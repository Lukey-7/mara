---
title: CAP theorem and consistency models
tags: [consistency, cap, theory]
published_date: 2026-09-02
---

# CAP theorem and consistency models

## The CAP theorem

Brewer's conjecture, proved by Gilbert and Lynch in 2002, states that a distributed data
store cannot simultaneously provide linearizable consistency, availability of every request,
and tolerance of network partitions. Because partitions cannot be prevented, the real choice
during a partition is between consistency (refuse or delay some requests) and availability
(answer, possibly with stale data). Outside partitions, the trade-off is latency versus
consistency, which is what the PACELC formulation adds.

## Common misreadings

CAP is not "pick two of three": partition tolerance is not optional in any system that spans a
network. Consistency in CAP means linearizability, not the C in ACID. Availability in CAP means
every non-failed node answers every request, which is stronger than the practical notion of
high availability with failover.

## Consistency models

Linearizability: every operation appears to take effect atomically at some point between its
start and end, in an order consistent with real time. Sequential consistency: operations appear
in some total order consistent with each client's program order, without the real-time
requirement. Causal consistency: writes that are causally related are seen in the same order
everywhere; concurrent writes may be seen in different orders. Eventual consistency: if writes
stop, all replicas converge to the same state; no bound on when.

## Read-your-writes and session guarantees

Session guarantees sit between causal and eventual: read-your-writes, monotonic reads,
monotonic writes and writes-follow-reads. They are cheap to provide by pinning a session to a
replica or by attaching a version vector to the session, and remove the most user-visible
anomalies of eventual consistency.

## Choosing a model

Systems like Spanner and CockroachDB pick consistency (CP) and pay with latency and reduced
availability during partitions. Dynamo-style stores pick availability (AP) and push conflict
resolution to the application. Many products offer tunable consistency per request, for
example quorum sizes in Cassandra or bounded staleness in Cosmos DB.
