# Relational & NoSQL Database Design

## Overview
Designed and implemented two database solutions for different business needs: a normalized relational schema for a transactional e-commerce business, and a MongoDB document schema for a health/fitness-tracking application with flexible, evolving fields.

## Relational design (MySQL)

**Context.** An online marketplace for sustainable products had been storing sales, product, and customer data in flat spreadsheets. That approach didn't scale, offered no data integrity guarantees, and made it slow to answer basic questions about product performance, regional trends, or channel mix.

**Schema.** The dataset was normalized into five related tables:

- `Products (ProductID, ItemType, UnitPrice, UnitCost)`
- `Customers (CustomerID, Country, Region)`
- `Channels (ChannelID, SalesChannel)`
- `Orders (OrderID, OrderDate, ShipDate, OrderPriority, CustomerID, ChannelID)`
- `Sales (SaleID, OrderID, ProductID, UnitsSold, TotalRevenue, TotalProfit)`

Relationships: a customer can place many orders; each order uses one sales channel; a product can appear across many sales transactions; the `Sales` table links orders to the specific products sold. Primary/foreign keys enforce referential integrity throughout.

**Implementation.** Raw CSV data was loaded into a staging table, validated, and normalized into the final schema above. Three business-insight queries were written and executed:

1. Total revenue by product type
2. Orders by region
3. Online vs. offline channel performance

**Performance.** Indexes were added on the columns most used for filtering and joins (order date, product ID, region). On a representative aggregate query, execution time dropped from ~383ms to ~327ms (roughly 15% faster) after indexing.

**Scalability & security.** The design anticipates growth through indexing, partitioning candidates for large tables (e.g., `Sales`), and a data-access layer that can support future API/cloud integration. Role-based access control, field-level encryption for sensitive customer data, and audit trails were included in the design to support privacy compliance (GDPR-style considerations).

## NoSQL design (MongoDB)

**Context.** A digital health company stored wearable-device and patient health data as unstructured JSON files. Without indexing, even simple lookups required full document scans, making it hard to flag underperforming devices or patients with health risks in anything close to real time.

**Design.** A document-oriented MongoDB schema was chosen specifically because the two source datasets had different, evolving shapes (some devices carry extra sensor fields; some patient records carry more health attributes than others) -- a fit for MongoDB's flexible, per-document schema rather than a rigid relational table. Each source dataset was mapped to its own collection:

- `fitness_trackers` -- one document per wearable device (battery life, customer rating, features)
- `medical` -- one document per patient record (vitals, known allergies, risk flags)

**Queries.** Three queries were written and evaluated:

1. Fitness trackers with battery life under 7 days (flags devices likely to frustrate customers)
2. Fitness trackers rated below 4.0 (flags underperforming products for review)
3. Patients with known allergies (flags higher-risk patients for custom care recommendations)

**Performance (before/after indexing):**

| Query | Docs examined before | Docs examined after | Query plan | Exec time |
|---|---|---|---|---|
| Battery life < 7 days | 565 | 190 | COLLSCAN → IXSCAN | -- |
| Rating < 4.0 | 565 | 95 | COLLSCAN → IXSCAN | 1ms → 0ms |
| Known allergies | 100,000 | 21,000 | COLLSCAN → IXSCAN | 176ms → 99ms |

**Scalability & security.** The design calls for indexing on the most frequently queried fields, horizontal scaling via sharding as data volume grows, and schema flexibility so new fields (e.g., a new sensor reading) can be added without downtime or migration. On the security side: role-based access control scoped to specific collections, encryption in transit and at rest, schema validation rules to keep documents well-formed, and audit logging -- aligned with typical healthcare-data handling practices (HIPAA-style considerations).

## Skills demonstrated
Relational modeling & normalization, MongoDB document design, SQL query writing and optimization, index performance benchmarking, privacy-aware data handling (HIPAA/GDPR considerations).
