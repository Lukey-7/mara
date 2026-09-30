---
title: Quorums, failure detection and membership
tags: [availability, quorum, gossip]
published_date: 2026-09-05
---

# Quorums, failure detection and membership

## Quorum reads and writes

With N replicas, a write acknowledged by W replicas and a read consulting R replicas overlap
when W + R > N, so the read sees the latest acknowledged write. Common settings are N=3,
W=2, R=2. Lowering W or R trades consistency for latency and availability. Even with a
strict quorum, concurrent writes, partial write failures and read-repair timing can still
produce stale reads, so quorums are not a substitute for consensus when linearizability is
required.

## Sloppy quorums and hinted handoff

When the home replicas of a key are unreachable, a sloppy quorum lets any reachable nodes
accept the write temporarily and hand it back (hinted handoff) once the home nodes return.
This maximises write availability but means W + R > N no longer guarantees overlap.

## Failure detection

A node cannot distinguish a crashed peer from a slow network. Heartbeats with a timeout are
the standard detector; a longer timeout means fewer false positives but slower failover.
The phi accrual failure detector (Cassandra, Akka) outputs a suspicion level based on the
distribution of past heartbeat intervals instead of a fixed timeout.

## Gossip protocols

Each node periodically exchanges its view of cluster state with a few random peers; information
spreads epidemically in O(log N) rounds. Gossip disseminates membership, load and schema
versions without a central coordinator. SWIM adds indirect probing so a single slow link does
not cause a false failure verdict.

## Coordination services

ZooKeeper, etcd and Consul are small, strongly consistent stores (ZAB or Raft inside) used for
leader election, distributed locks with fencing tokens, configuration and service discovery.
Fencing tokens matter: a lock holder that pauses (GC, network) may believe it still holds the
lock; a monotonically increasing token checked by the storage layer rejects its stale writes.
