// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The entry reads the trial brief (provider directory, sealed quality table,
// workload manifest), probes every provider's catalog route, registers one
// catalog with the env, then serves the client's 20 requests: each answered
// with the best free model that meets the request's requirements, failing
// over to the next best when a provider refuses, and translated back into
// the wire the client used (OpenAI Chat Completions or Anthropic Messages,
// JSON or server-sent events).
//
// Routing: candidates are every free model meeting the request, ranked by
// the quality table's rating_milli. A provider that has refused twice is
// pushed to the back (its budget is likely spent or it is in outage), and
// the least-called provider wins a rating tie, so the load spreads over the
// per-trial budgets instead of draining the single best provider. One
// provider call at a time, in the order the client sent the requests, so a
// replay sees the same calls in the same order. No clock, no randomness:
// every choice is a function of the brief, the requests and the providers'
// answers so far.
//
// `node main.ts --list` prints the catalog and exits 0, the same document
// the register body carries, byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIAnswer,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The anthropic-version header the Anthropic wire expects. */
const ANTHROPIC_VERSION = '2023-06-01';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Provider calls made so far this trial, by provider id (a refusal counts: it spends budget too). */
const callsMade = new Map<string, number>();
/** Providers that refused twice in a row: their budget is likely spent or they are in outage. */
const coldProviders = new Set<string>();
/** Consecutive refusals per provider, across requests (a success resets it). */
const consecutiveFails = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered: what GET /v1/models lists, each id once. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can serve one request, best first. A provider that
 * refused twice in a row goes to the back: its budget is likely spent or it
 * is in outage, and retrying it first only burns calls. A rating tie goes to
 * the provider called least so far, which spreads the load over the budgets.
 * An Anthropic-wire provider cannot carry an OpenAI client's JSON mode (its
 * wire echoes json_mode false), so it never serves one.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  const providerOrder = new Map(brief.directory.providers.map((p, i) => [p.id, i]));
  return out.sort((a, b) => {
    const aCold = coldProviders.has(a.probe.provider.id) ? 1 : 0;
    const bCold = coldProviders.has(b.probe.provider.id) ? 1 : 0;
    if (aCold !== bCold) return aCold - bCold;
    if (a.rating !== b.rating) return b.rating - a.rating;
    const aCalls = callsMade.get(a.probe.provider.id) ?? 0;
    const bCalls = callsMade.get(b.probe.provider.id) ?? 0;
    if (aCalls !== bCalls) return aCalls - bCalls;
    const aOrder = providerOrder.get(a.probe.provider.id) ?? 0;
    const bOrder = providerOrder.get(b.probe.provider.id) ?? 0;
    if (aOrder !== bOrder) return aOrder - bOrder;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this client body asks for JSON mode (only the OpenAI wire can carry it). */
function jsonModeOf(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The upstream request body for one candidate, in the provider's wire. */
function upstreamBody(clientApi: 'openai' | 'anthropic', candidate: Candidate, body: Json, stream: boolean): Json {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (wire === 'openai') {
    if (clientApi === 'openai') {
      const out: Json = { ...body, model };
      if (stream) out.stream_options = { include_usage: true };
      return out;
    }
    return anthropicToOpenAI(body, model);
  }
  if (wire === 'anthropic') {
    if (clientApi === 'openai') return openAIToAnthropicRequest(body, model);
    return { ...body, model };
  }
  return clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
}

/** One inference call to one candidate's provider, in its wire. */
async function callProvider(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<Response> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let payload: Json;
  if (provider.wire === 'gemini') {
    const method = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${method}${stream ? '?alt=sse' : ''}`;
    payload = upstreamBody(clientApi, candidate, body, stream);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = ANTHROPIC_VERSION;
    payload = upstreamBody(clientApi, candidate, body, stream);
  } else {
    url = `${base}/v1/chat/completions`;
    payload = upstreamBody(clientApi, candidate, body, stream);
  }
  if (stream) headers.accept = 'text/event-stream';
  return fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
  });
}

/**
 * One provider's answer translated into the client's wire. JSON answers are
 * rebuilt; streams are translated event by event. The provider's text is
 * returned whole, its usage counts as its wire reports them, and its stop
 * reason mapped to the client's family.
 */
async function translateAnswer(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  stream: boolean,
  upstream: Response,
  res: ServerResponse,
): Promise<void> {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (!stream) {
    const answer: unknown = await upstream.json().catch(() => null);
    if (!isRecord(answer)) {
      sendError(res, clientApi, 502, 'upstream_bad_answer', 'the provider answered with a body that is not one JSON object');
      return;
    }
    if (wire === 'openai') {
      if (clientApi === 'openai') sendJson(res, 200, answer);
      else sendJson(res, 200, openAIToAnthropic(answer, model));
    } else if (wire === 'anthropic') {
      if (clientApi === 'openai') sendJson(res, 200, anthropicToOpenAIAnswer(answer, model));
      else sendJson(res, 200, { ...answer, model });
    } else if (clientApi === 'openai') sendJson(res, 200, geminiToOpenAI(answer, model));
    else sendJson(res, 200, geminiToAnthropic(answer, model));
    return;
  }
  // Streaming.
  if (wire === 'openai' && clientApi === 'openai') {
    // Same wire both sides: pipe the provider's events straight through.
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (upstream.body === null) {
      res.end('data: [DONE]\n\n');
      return;
    }
    const reader = upstream.body.getReader();
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        res.write(Buffer.from(value));
      }
    } finally {
      reader.releaseLock();
      res.end();
    }
    return;
  }
  if (wire === 'anthropic' && clientApi === 'anthropic') {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (upstream.body === null) {
      res.end();
      return;
    }
    const reader = upstream.body.getReader();
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        res.write(Buffer.from(value));
      }
    } finally {
      reader.releaseLock();
      res.end();
    }
    return;
  }
  const text = await upstream.text();
  if (wire === 'openai') {
    // OpenAI upstream, Anthropic client: run the provider's events through the
    // same translator the streaming path uses, fed from the buffered text.
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    const bytes = new TextEncoder().encode(text);
    const web = new ReadableStream<Uint8Array>({ start(c) { c.enqueue(bytes); c.close(); } });
    for await (const event of anthropicEventsFromOpenAI(web, model)) res.write(event);
    res.end();
    return;
  }
  if (wire === 'anthropic') {
    sendSse(res, openAIEventsFromAnthropic(text, model));
    return;
  }
  if (clientApi === 'openai') sendSse(res, openAIEventsFromGemini(text, model));
  else sendSse(res, anthropicEventsFromGemini(text, model));
}

/**
 * Serve one client request: try the candidates best first, and on a refusal
 * (402, 429, 5xx, unreachable, timed out) answer it with the next one. A
 * refusal on a provider's failure schedule spends that one call number, so
 * the next best model serves at nearly the same rating; two refusals in a
 * row mean the budget is spent or the provider is in outage, so it goes
 * cold for the rest of the trial. No immediate retry: quality is already at
 * its ceiling (only one model rates 1475 and its budget cannot cover all
 * seventeen requests that want it), and a retry would only cost a call the
 * tie-break counts. Exactly one provider call may answer a request: the loop
 * stops at the first success. A last pass over the best three retries them
 * once more if nothing served yet.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires, clientApi === 'openai' && jsonModeOf(body));
  if (candidates.length === 0) {
    sendError(res, clientApi, 503, 'no_free_model', 'no free model on the directory meets this request');
    return;
  }
  let lastStatus = 0;
  const passes: Candidate[][] = [candidates, candidates.slice(0, 3)];
  for (let pass = 0; pass < passes.length; pass++) {
    for (const candidate of passes[pass]) {
      const providerId = candidate.probe.provider.id;
      const key = `${providerId}/${candidate.model.id}`;
      callsMade.set(providerId, (callsMade.get(providerId) ?? 0) + 1);
      let upstream: Response;
      try {
        upstream = await callProvider(candidate, clientApi, body, stream);
      } catch (err) {
        const detail = (err as Error).message;
        log(`refused ${key}: unreachable (${detail.slice(0, 120)})`);
        noteRefusal(providerId);
        continue;
      }
      if (upstream.ok) {
        consecutiveFails.set(providerId, 0);
        log(`served by ${key}`);
        await translateAnswer(candidate, clientApi, stream, upstream, res);
        return;
      }
      lastStatus = upstream.status;
      let detail = '';
      try {
        detail = (await upstream.text()).slice(0, 200);
      } catch {
        detail = '';
      }
      log(`refused ${key}: HTTP ${upstream.status} ${detail.slice(0, 120)}`);
      noteRefusal(providerId);
      // A 400/401 is this entry's own mistake for that provider (a model it
      // does not serve, a wrong key): the next candidate is the fix, and the
      // provider takes no more of this trial's traffic.
      if (upstream.status === 400 || upstream.status === 401) coldProviders.add(providerId);
    }
  }
  sendError(res, clientApi, 503, 'no_provider_served', `every free model refused this request (last HTTP ${lastStatus})`);
}

/** Record one refusal: two in a row send the provider cold. */
function noteRefusal(providerId: string): void {
  const count = (consecutiveFails.get(providerId) ?? 0) + 1;
  consecutiveFails.set(providerId, count);
  if (count >= 2) coldProviders.add(providerId);
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
