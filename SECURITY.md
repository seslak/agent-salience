# Security Policy

Agent Salience is a local, stdlib-only Python library. It does not make network calls, execute shell commands, store secrets, or contact external services.

## Supported versions

Security fixes are expected for the latest released version.

## Reporting a vulnerability

If you find a security issue, please open a private advisory on GitHub if available, or contact the maintainer through the repository owner profile.

Please include:

- affected version or commit
- minimal reproduction
- expected vs actual behavior
- potential impact

## Scope

In scope:

- unsafe parsing behavior
- unexpected file/network/process access
- denial-of-service risks from unbounded local computation
- serialization/deserialization issues

Out of scope:

- caller policy mistakes
- misuse of salience scores by downstream agents
- model behavior from external LLM/embedding systems
- secrets accidentally placed in caller-provided text

## Data handling

Agent Salience processes caller-provided text in memory and returns scores/signatures. It does not persist data by itself.
