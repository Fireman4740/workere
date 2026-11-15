# MAVEN200 MAXCTX on one H100

User msg_f3ced25e requests the maximum context. The pinned checkpoint's
text_config.max_position_embeddings is262144,with default RoPE and no scaling.
The model's native window is used without altering its weights or RoPE.

Parent D established the current reference recipe: same200documents,five fixed
training demonstrations/readings,greedy decoding,4096-token JSON answers,and
majority>=3/5. MAXCTX keeps those choices and raises the reasoning ceiling to
253952tokens,leaving8192tokens for prompt,answer and safety. Every one of the
1000 actual tokenized prompts must fit; none may be truncated or excluded.
Thinking can stop naturally before its ceiling.

One H100NVL,context262144,concurrency4,unchanged SIF,checkpoint,FP8KV and
MLPint8. Existing H100 server logs show2,138,202KV tokens for concurrency32;
four full native windows need1,048,576tokens. The actual new configuration
must pass a synthetic262016-token prefill and four simultaneous short schema
checks before model evaluation. No evaluation labels enter these preflights.

Primary: paired document-bootstrap micro-F1 difference versus frozen D,
5000draws,seed20261112. P/R,inter/intra,distance bins,forced thinking,
truncation,invalid answers,tokens and durations are retained. This is an
additional exposed-development comparison; the H100/concurrency4 runtime
differs from cached D's L40S/concurrency32,so a causal context-only claim is
not justified. No final score is produced for an incomplete panel.

12h GPU allocations; up to5h scheduling new work per allocation. Thinking
requests have a5h timeout; final JSON requests20min. The absolute Slurm end
time imposes an additional5h25m drain guard,including final-response time
and cleanup. Each trace and response is saved atomically. Checkpoint exit75
or parent SlurmTIMEOUT triggers another allocation,which only runs missing
work. Maximum24allocations,up to120h of scheduled work; CPUcontroller14days.
Unrelated errors and user cancellations are not retried blindly.

Outputs: /sps/liris/mcarlesso/maven-maxctx-20261115/MAXCTX.
Fixed inherited entrypoint: bash steering/run_pilot.sh.
