# Schema Profile

**Database:** `/Users/mdev/dev/sqlprobe/data/tpcds.db`
**Tables:** 8
**Total columns:** 61
**Naming ambiguity score:** 0.289

## Tables

### customer (150,000 rows)

**Primary key:** c_custkey

| Column | Type | Nullable | PK |
|--------|------|----------|----|
| c_custkey | BIGINT | NO | PK |
| c_name | VARCHAR | NO |  |
| c_address | VARCHAR | NO |  |
| c_nationkey | INTEGER | NO |  |
| c_phone | VARCHAR | NO |  |
| c_acctbal | DECIMAL(15,2) | NO |  |
| c_mktsegment | VARCHAR | NO |  |
| c_comment | VARCHAR | NO |  |

### lineitem (6,001,215 rows)

**Primary key:** l_orderkey, l_linenumber

| Column | Type | Nullable | PK |
|--------|------|----------|----|
| l_orderkey | BIGINT | NO | PK |
| l_partkey | BIGINT | NO |  |
| l_suppkey | BIGINT | NO |  |
| l_linenumber | BIGINT | NO | PK |
| l_quantity | DECIMAL(15,2) | NO |  |
| l_extendedprice | DECIMAL(15,2) | NO |  |
| l_discount | DECIMAL(15,2) | NO |  |
| l_tax | DECIMAL(15,2) | NO |  |
| l_returnflag | VARCHAR | NO |  |
| l_linestatus | VARCHAR | NO |  |
| l_shipdate | DATE | NO |  |
| l_commitdate | DATE | NO |  |
| l_receiptdate | DATE | NO |  |
| l_shipinstruct | VARCHAR | NO |  |
| l_shipmode | VARCHAR | NO |  |
| l_comment | VARCHAR | NO |  |

### nation (25 rows)

**Primary key:** n_nationkey

| Column | Type | Nullable | PK |
|--------|------|----------|----|
| n_nationkey | INTEGER | NO | PK |
| n_name | VARCHAR | NO |  |
| n_regionkey | INTEGER | NO |  |
| n_comment | VARCHAR | NO |  |

### orders (1,500,000 rows)

**Primary key:** o_orderkey

| Column | Type | Nullable | PK |
|--------|------|----------|----|
| o_orderkey | BIGINT | NO | PK |
| o_custkey | BIGINT | NO |  |
| o_orderstatus | VARCHAR | NO |  |
| o_totalprice | DECIMAL(15,2) | NO |  |
| o_orderdate | DATE | NO |  |
| o_orderpriority | VARCHAR | NO |  |
| o_clerk | VARCHAR | NO |  |
| o_shippriority | INTEGER | NO |  |
| o_comment | VARCHAR | NO |  |

### part (200,000 rows)

**Primary key:** p_partkey

| Column | Type | Nullable | PK |
|--------|------|----------|----|
| p_partkey | BIGINT | NO | PK |
| p_name | VARCHAR | NO |  |
| p_mfgr | VARCHAR | NO |  |
| p_brand | VARCHAR | NO |  |
| p_type | VARCHAR | NO |  |
| p_size | INTEGER | NO |  |
| p_container | VARCHAR | NO |  |
| p_retailprice | DECIMAL(15,2) | NO |  |
| p_comment | VARCHAR | NO |  |

### partsupp (800,000 rows)

**Primary key:** ps_partkey, ps_suppkey

| Column | Type | Nullable | PK |
|--------|------|----------|----|
| ps_partkey | BIGINT | NO | PK |
| ps_suppkey | BIGINT | NO | PK |
| ps_availqty | BIGINT | NO |  |
| ps_supplycost | DECIMAL(15,2) | NO |  |
| ps_comment | VARCHAR | NO |  |

### region (5 rows)

**Primary key:** r_regionkey

| Column | Type | Nullable | PK |
|--------|------|----------|----|
| r_regionkey | INTEGER | NO | PK |
| r_name | VARCHAR | NO |  |
| r_comment | VARCHAR | NO |  |

### supplier (10,000 rows)

**Primary key:** s_suppkey

| Column | Type | Nullable | PK |
|--------|------|----------|----|
| s_suppkey | BIGINT | NO | PK |
| s_name | VARCHAR | NO |  |
| s_address | VARCHAR | NO |  |
| s_nationkey | INTEGER | NO |  |
| s_phone | VARCHAR | NO |  |
| s_acctbal | DECIMAL(15,2) | NO |  |
| s_comment | VARCHAR | NO |  |

## Foreign Keys

| From | To |
|------|----|
| customer.c_nationkey | nation.n_nationkey |
| lineitem.l_orderkey | orders.o_orderkey |
| lineitem.l_partkey | part.p_partkey |
| lineitem.l_suppkey | supplier.s_suppkey |
| nation.n_regionkey | region.r_regionkey |
| orders.o_custkey | customer.c_custkey |
| partsupp.ps_partkey | part.p_partkey |
| partsupp.ps_suppkey | supplier.s_suppkey |
| supplier.s_nationkey | nation.n_nationkey |

## Naming Ambiguity Analysis

Score: **0.289** (0.0 = no ambiguity, 1.0 = high ambiguity)

This measures what fraction of column base-names (after stripping table prefixes) appear in multiple tables. Higher scores mean LLMs must work harder to disambiguate which table a column belongs to.
