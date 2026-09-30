---
title: Distributed transactions
tags: [transactions, 2pc, consistency]
published_date: 2026-09-03
---

# Distributed transactions

A transaction that touches several partitions or services must commit atomically on all of
them or none. The classic protocol is two-phase commit; modern databases layer it over
consensus to remove its single point of failure.

## Two-phase commit (2PC)

A coordinator asks every participant to prepare: each writes its intent durably and votes yes
or no. If all vote yes, the coordinator records a commit decision and tells participants to
commit; any no (or timeout) leads to abort. Once a participant votes yes it must hold its locks
and wait for the decision. If the coordinator crashes after participants prepared, they are
blocked until it recovers: 2PC is a blocking protocol. Three-phase commit adds a pre-commit
step to avoid blocking, but assumes bounded delays and is rarely used.

## 2PC over consensus

Spanner and CockroachDB make the coordinator state itself a replicated log (Paxos or Raft
group), so losing one coordinator node does not block participants. Each participant is also a
consensus group. The cost is several round trips of consensus per transaction.

## Isolation across partitions

Serializable isolation across shards needs either two-phase locking with distributed deadlock
detection, or an ordering mechanism such as globally synchronised timestamps (Spanner's
TrueTime) or hybrid logical clocks (CockroachDB). Snapshot isolation is commonly offered
instead: reads see a consistent snapshot as of a timestamp and writes are checked for conflicts
at commit.

## Sagas

When locking across services is unacceptable, a saga runs a sequence of local transactions,
each with a compensating action. If step k fails, compensations for steps k−1 … 1 run in
reverse. Sagas give atomicity in the sense of "eventually all or none" but not isolation:
other transactions can observe intermediate states.

## Idempotency and exactly-once

Retries after a timeout can apply an operation twice. Making operations idempotent with a
client-supplied request id, deduplicated at the receiver, is what "exactly-once" processing
means in practice: at-least-once delivery plus idempotent application.
