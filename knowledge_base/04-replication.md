---
title: Replication strategies
tags: [replication, availability]
published_date: 2026-09-02
---

# Replication strategies

Replication keeps copies of data on several nodes for fault tolerance, read scaling and
geographic locality. The three broad designs are single-leader, multi-leader and leaderless.

## Single-leader replication

One replica accepts writes and streams a change log to followers. Followers serve reads.
Synchronous followers must acknowledge before the write is confirmed, giving durability at the
cost of latency; asynchronous followers lag and can lose recent writes on failover. Most
systems use one synchronous follower plus asynchronous ones (semi-synchronous). Failover has
to detect the leader failure, choose a new leader with the most complete log, and reconfigure
clients; split brain occurs if two nodes both believe they are leader.

## Replication log formats

Statement-based replication ships SQL statements and breaks on non-deterministic functions.
Write-ahead-log shipping sends the physical log and ties leader and follower to the same
storage version. Logical (row-based) replication sends row changes and decouples versions;
it is also the basis for change data capture.

## Multi-leader replication

Several data centres each have a leader; leaders replicate to each other asynchronously. Local
writes are fast and each site survives losing the others, but concurrent writes to the same
record conflict and must be resolved: last-writer-wins by timestamp, merge by application code,
or conflict-free replicated data types (CRDTs).

## Leaderless replication

Clients (or a coordinator) send writes to all N replicas and consider the write successful once
W acknowledge; reads query R replicas and take the newest version. With W + R > N a read is
guaranteed to overlap at least one replica holding the latest write. Read repair and anti-entropy
processes bring stale replicas up to date. Dynamo, Cassandra and Riak use this model, together
with sloppy quorums and hinted handoff during failures.

## Replication lag anomalies

Asynchronous replication exposes users to reading their own missing write, seeing time go
backwards between two reads (non-monotonic reads), and seeing an answer before its question
(violated causality). Session guarantees or causal consistency address these.
