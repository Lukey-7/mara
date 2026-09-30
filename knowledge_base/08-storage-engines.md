---
title: Storage engines - LSM trees and B-trees
tags: [storage, lsm, btree]
published_date: 2026-09-04
---

# Storage engines: LSM trees and B-trees

## B-trees

A B-tree stores sorted key ranges in fixed-size pages (usually 4 KB) arranged as a balanced
tree with high fan-out, so a lookup touches only a few pages. Updates modify pages in place;
a write-ahead log protects against crashes mid-update. B-trees give predictable read latency
and are the default in PostgreSQL, MySQL InnoDB and most relational engines.

## Log-structured merge trees

Writes go to an in-memory sorted structure (the memtable) and to a write-ahead log. When the
memtable fills, it is flushed to disk as an immutable sorted file (an SSTable). Reads check the
memtable, then SSTables from newest to oldest. Background compaction merges SSTables, drops
overwritten and deleted keys (tombstones), and keeps the number of files bounded. LSM trees
are used by LevelDB, RocksDB, Cassandra and HBase.

## Compaction strategies

Size-tiered compaction merges files of similar size; it has low write amplification but can
leave many overlapping files, hurting reads and space. Levelled compaction keeps each level as
non-overlapping files roughly ten times larger than the previous level; reads touch one file per
level, at the cost of more rewriting.

## Read optimisations

Bloom filters per SSTable let a read skip files that certainly do not contain the key. Sparse
in-memory indexes point into each SSTable so a lookup reads one block. Block caches keep hot
data in memory.

## Trade-offs

LSM trees turn random writes into sequential ones and compress well, so they win on write-heavy
workloads and SSDs. B-trees avoid compaction stalls, give each key one location (simpler
transactional locking) and are faster for point reads on read-heavy workloads. Write
amplification (bytes written to disk per byte of user data) and space amplification are the
metrics used to compare them.
