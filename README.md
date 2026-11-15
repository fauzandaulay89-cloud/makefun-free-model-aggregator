# makefun-free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/fauzandaulay89-cloud/makefun-free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/fauzandaulay89-cloud/makefun-free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A free AI model aggregator: one server that answers the OpenAI Chat Completions API and the Anthropic Messages API — JSON and server-sent events — by routing every request to the best free model across ten providers, failing over to the next best when a provider refuses, and translating the answer back into the wire the client used. It registers its catalog with the trial env, serves `GET /v1/models` in either wire's shape, and prints the same catalog with `--list`.

## Run it

```sh
node main.ts --list   # print the model catalog and exit
node main.ts          # serve OpenAI + Anthropic APIs (needs MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL from a trial)
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=fxram5cz&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
