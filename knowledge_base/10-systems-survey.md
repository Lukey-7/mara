---
title: Survey of distributed database systems
tags: [systems, survey]
published_date: 2026-09-05
---

# Survey of distributed database systems

## Google Bigtable and HBase

Bigtable is a sparse, sorted map from (row, column, timestamp) to value, partitioned into
tablets by row range and stored on GFS as SSTables with an LSM design. A single master assigns
tablets; Chubby (Paxos) holds the master lock and the root tablet location. HBase is the
open-source equivalent on HDFS with ZooKeeper.

## Amazon Dynamo and Cassandra

Dynamo introduced the leaderless, always-writable design: consistent hashing with virtual
nodes, sloppy quorums, hinted handoff, vector clocks for conflict detection and Merkle trees for
anti-entropy. Cassandra combines Dynamo's ring with Bigtable's data model and LSM storage,
replaces vector clocks with last-writer-wins timestamps per cell, and adds tunable consistency
per query.

## Google Spanner

Spanner is a globally distributed SQL database with external consistency. Data is split into
tablets replicated by Paxos groups across data centres; transactions use two-phase commit over
those groups, two-phase locking and TrueTime-assigned timestamps. Read-only transactions read a
snapshot at a chosen timestamp without locks.

## CockroachDB and TiDB

Both are open-source, Spanner-inspired SQL databases. CockroachDB stores ranges in RocksDB (later
Pebble), replicates each range with Raft, and orders transactions with hybrid logical clocks and
a per-transaction record. TiDB separates the SQL layer from TiKV, a Raft-replicated key-value
store, and offers a columnar replica (TiFlash) for analytics.

## MongoDB

MongoDB shards collections by a shard key and replicates each shard with a replica set (one
primary, elected via a Raft-like protocol). Writes go to the primary; read preference lets
clients read from secondaries with eventual consistency. Multi-document transactions arrived in
4.0 and became cross-shard in 4.2.

## Choosing between them

Strong consistency and SQL across regions: Spanner, CockroachDB, TiDB, at the cost of
cross-region commit latency. Write availability and simple key-based access: Cassandra,
DynamoDB. Flexible documents with a single-primary model: MongoDB. Wide-column analytics at
scale on a Hadoop stack: HBase.
