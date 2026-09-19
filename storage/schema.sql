-- ============================================================
-- SCHEMA: stock analysis platform
-- Compatible con Supabase (PostgreSQL 15+)
-- ============================================================

-- Extensiones necesarias
create extension if not exists "uuid-ossp";


-- ============================================================
-- TABLA: stocks
-- Catálogo de todos los tickers conocidos por el sistema.
-- Se rellena una vez y se actualiza raramente.
-- ============================================================
create table if not exists stocks (
    ticker          varchar(10)     primary key,
    name            varchar(255)    not null,
    exchange        varchar(20),                        -- NYSE, NASDAQ, etc.
    sector          varchar(100),
    industry        varchar(150),
    is_active       boolean         not null default true,
    created_at      timestamptz     not null default now()
);

comment on table  stocks              is 'Catálogo maestro de tickers.';
comment on column stocks.ticker       is 'Símbolo bursátil. PK natural (AAPL, MSFT…).';
comment on column stocks.is_active    is 'False si el ticker fue delistado o descartado.';


-- ============================================================
-- TABLA: stock_prices
-- Serie temporal de precios OHLCV. Append-only.
-- Una fila por ticker × día de mercado.
-- ============================================================
create table if not exists stock_prices (
    id              bigint          generated always as identity primary key,
    ticker          varchar(10)     not null references stocks(ticker) on delete cascade,
    price_date      date            not null,
    open            numeric(12, 4)  not null,
    high            numeric(12, 4)  not null,
    low             numeric(12, 4)  not null,
    close           numeric(12, 4)  not null,
    adj_close       numeric(12, 4),                     -- ajustado por splits/dividendos
    volume          bigint,
    inserted_at     timestamptz     not null default now(),

    constraint uq_stock_prices_ticker_date unique (ticker, price_date)
);

comment on table  stock_prices              is 'Precios OHLCV diarios. Una fila por ticker×día.';
comment on column stock_prices.adj_close    is 'Precio ajustado. Usa este para cálculos de retorno.';
comment on column stock_prices.inserted_at  is 'Momento en que el script insertó la fila (auditoría).';

-- Índices de consulta habitual
create index if not exists idx_sp_ticker_date
    on stock_prices (ticker, price_date desc);

create index if not exists idx_sp_date
    on stock_prices (price_date desc);


-- ============================================================
-- TABLA: signals
-- Resultados del algoritmo de detección. Una fila por
-- ticker × día en que el algoritmo lo marcó como señal.
-- ============================================================
create table if not exists signals (
    id              bigint          generated always as identity primary key,
    ticker          varchar(10)     not null references stocks(ticker) on delete cascade,
    signal_date     date            not null,
    score           numeric(5, 4),                      -- 0.0000 – 1.0000
    action          varchar(10)     not null default 'buy'
                        check (action in ('buy', 'sell', 'watch')),
    indicators      jsonb,                              -- valores intermedios del algoritmo
    created_at      timestamptz     not null default now(),

    constraint uq_signals_ticker_date unique (ticker, signal_date)
);

comment on table  signals            is 'Señales generadas por el algoritmo cada día.';
comment on column signals.score      is 'Confianza de la señal (1 = máxima).';
comment on column signals.indicators is 'JSON con los indicadores calculados (RSI, MACD, etc.).';

create index if not exists idx_signals_date
    on signals (signal_date desc);

create index if not exists idx_signals_ticker_date
    on signals (ticker, signal_date desc);

-- Índice GIN para consultas dentro del JSON de indicadores
create index if not exists idx_signals_indicators
    on signals using gin (indicators);


-- ============================================================
-- TABLA: users
-- Gestionada por Supabase Auth + esta tabla de perfil.
-- El id debe coincidir con auth.users.id de Supabase.
-- ============================================================
create table if not exists users (
    id              uuid            primary key default uuid_generate_v4(),
    email           varchar(255)    not null unique,
    plan            varchar(20)     not null default 'free'
                        check (plan in ('free', 'premium', 'admin')),
    plan_expires_at timestamptz,                        -- null = sin expiración (admin/vitalicio)
    created_at      timestamptz     not null default now()
);

comment on table  users                 is 'Perfiles de usuario vinculados a Supabase Auth.';
comment on column users.plan            is 'free = acceso a señales públicas. premium = análisis custom.';
comment on column users.plan_expires_at is 'Si es null y plan=premium, no caduca (pago único o admin).';


-- ============================================================
-- TABLA: hmm_results
-- Almacena los resultados del modelo HMM (Hidden Markov Model)
-- para análisis de regímenes de volatilidad.
-- ============================================================
DROP TABLE IF EXISTS hmm_results;

CREATE TABLE hmm_results (
    id                      bigint          generated always as identity primary key,
    ticker                  varchar(10)     not null,
    result_date             date            not null,
    regime_label            varchar(20)     not null,
    state_id                integer         not null,
    probability             numeric(12, 8)  not null,
    state_probabilities     jsonb           not null,
    volatility_level        varchar(20)     not null,
    position_multiplier     numeric(12, 8)  not null,
    is_confirmed            boolean         not null default false,
    is_flickering           boolean         not null default false,
    stability_bars          integer         not null,
    bic_score               numeric          not null,
    n_regimes               integer          not null,
    training_date           timestamptz      not null,
    created_at              timestamptz      not null default now(),
    constraint uq_hmm_results_ticker_date unique (ticker, result_date)
);

CREATE INDEX IF NOT EXISTS idx_hmm_results_two_layer_ticker_date
    ON hmm_results (ticker, result_date desc);

CREATE INDEX IF NOT EXISTS idx_hmm_results_two_layer_date
    ON hmm_results (result_date desc);


-- ============================================================
-- TABLA: hmm_market_results
-- One persisted macro regime per market index and market date.
-- ============================================================
CREATE TABLE IF NOT EXISTS hmm_market_results (
    id                      bigint generated always as identity primary key,
    market_ticker           varchar(10) not null,
    result_date             date not null,
    regime_label            varchar(20) not null,
    state_id                integer not null,
    probability             numeric(12, 8) not null,
    state_probabilities     jsonb not null,
    base_multiplier         numeric(12, 8) not null,
    is_regime_confirmed     boolean not null,
    is_flickering           boolean not null,
    stability_bars          integer not null,
    bic_score               numeric,
    n_states                integer not null,
    trained_at              timestamptz not null,
    created_at              timestamptz not null default now(),
    constraint uq_hmm_market_results_ticker_date unique (market_ticker, result_date)
);

CREATE INDEX IF NOT EXISTS idx_hmm_market_results_date
    ON hmm_market_results (result_date desc);


-- ============================================================
-- TABLA: custom_requests
-- Solicitudes de análisis premium: un usuario pide
-- que el algoritmo corra sobre un ticker específico.
-- ============================================================
create table if not exists custom_requests (
    id              bigint          generated always as identity primary key,
    user_id         uuid            not null references users(id) on delete cascade,
    ticker          varchar(10)     not null references stocks(ticker) on delete restrict,
    status          varchar(20)     not null default 'pending'
                        check (status in ('pending', 'running', 'done', 'error')),
    result          jsonb,                              -- output del algoritmo (igual que signals.indicators)
    error_message   text,                              -- descripción del error si status = 'error'
    requested_at    timestamptz     not null default now(),
    completed_at    timestamptz
);

comment on table  custom_requests              is 'Cola de análisis on-demand del plan premium.';
comment on column custom_requests.status       is 'pending → running → done | error.';
comment on column custom_requests.result       is 'Mismo esquema JSON que signals.indicators.';

create index if not exists idx_cr_user_status
    on custom_requests (user_id, status, requested_at desc);

create index if not exists idx_cr_status
    on custom_requests (status) where status in ('pending', 'running');


-- Two-layer regime architecture persistence.
-- Layer 1 model artifacts may remain as JSON files; this migration stores the
-- per-stock model contract and every final allocation result for auditability.

create table if not exists stock_risk_models (
    ticker text not null,
    model_version text not null,
    schema_version integer not null default 1,
    market_model_version text,
    model_type text not null,
    created_at timestamptz not null,
    config jsonb not null,
    primary key (ticker, model_version)
);

create index if not exists idx_stock_risk_models_ticker_created
    on public.stock_risk_models (ticker, created_at desc);

create table if not exists allocation_results (
    id bigint generated by default as identity primary key,
    result_timestamp timestamptz not null,
    ticker text not null,
    market_model_version text not null references market_regime_models(model_version),
    stock_model_version text not null,
    market_regime text not null,
    market_regime_probability double precision not null,
    base_multiplier double precision not null,
    relative_vol double precision not null,
    beta double precision not null,
    vol_scalar double precision not null,
    final_multiplier double precision not null,
    is_regime_confirmed boolean not null,
    is_flickering boolean not null,
    reasoning text not null,
    created_at timestamptz not null default now(),
    unique (ticker, result_timestamp, market_model_version, stock_model_version)
);

create table if not exists market_regime_models (
    model_version text primary key,
    model_type text not null,
    schema_version integer not null default 1,
    market_ticker text not null,
    training_date timestamptz not null,
    n_states integer not null,
    bic_score double precision not null,
    artifact jsonb not null,
    status text not null default 'ACTIVE',
    created_at timestamptz not null default now()
);

create unique index if not exists uq_market_regime_models_active_ticker
    on market_regime_models (market_ticker)
    where status = 'ACTIVE';

create table if not exists market_regime_results (
    id bigint generated by default as identity primary key,
    result_date date not null,
    market_ticker text not null,
    market_model_version text not null references market_regime_models(model_version),
    state_id integer not null,
    regime_label text not null,
    probability double precision not null,
    state_probabilities jsonb not null,
    base_multiplier double precision not null,
    is_confirmed boolean not null,
    consecutive_bars integer not null,
    is_flickering boolean not null,
    created_at timestamptz not null default now(),
    unique (market_ticker, result_date, market_model_version)
);

create index if not exists idx_market_regime_results_date
    on market_regime_results (result_date desc);

create index if not exists idx_allocation_results_ticker_timestamp
    on public.allocation_results (ticker, result_timestamp desc);

create index if not exists idx_allocation_results_timestamp
    on public.allocation_results (result_timestamp desc);

create index if not exists idx_allocation_results_timestamp
    on allocation_results (result_timestamp desc);

alter table stock_risk_models
    add column if not exists model_type text;
alter table stock_risk_models
    add column if not exists schema_version integer default 1;
alter table stock_risk_models
    add column if not exists market_model_version text;
alter table stock_risk_models
    add column if not exists created_at timestamptz default now();
alter table stock_risk_models
    add column if not exists config jsonb;

alter table allocation_results
    add column if not exists market_model_version text;
alter table allocation_results
    add column if not exists stock_model_version text;

alter table allocation_results
    drop constraint if exists allocation_results_unique_daily;

-- If allocation_results already existed before this migration, the new
-- uniqueness constraint is added separately so retries remain idempotent.
create unique index if not exists uq_allocation_results_logical
    on allocation_results (ticker, result_timestamp, market_model_version, stock_model_version);


-- ============================================================
-- ROW LEVEL SECURITY (RLS)
-- Habilitar en todas las tablas de negocio.
-- ============================================================

alter table stocks          enable row level security;
alter table stock_prices    enable row level security;
alter table signals         enable row level security;
alter table hmm_results    enable row level security;
alter table hmm_market_results enable row level security;
alter table users           enable row level security;
alter table custom_requests enable row level security;
alter table allocation_results enable row level security;
alter table stock_risk_models enable row level security;
alter table market_regime_models enable row level security;
alter table market_regime_results enable row level security;

-- stocks: lectura pública (cualquiera puede ver el catálogo)
create policy "stocks_select_public"
    on stocks for select
    using (true);

-- stocks: inserción pública (para scripts de carga como ticker_loader.py)
create policy "stocks_insert_public"
    on stocks for insert
    with check (true);

-- stock_prices: lectura pública
create policy "sp_select_public"
    on stock_prices for select
    using (true);

-- stock_prices: inserción pública (para scripts de carga como data_loader_historical.py)
create policy "sp_insert_public"
    on stock_prices for insert
    with check (true);

-- signals: lectura pública (la parte gratuita de la web)
create policy "signals_select_public"
    on signals for select
    using (true);

-- hmm_results: lectura pública
create policy "hmm_results_select_public"
    on hmm_results for select
    using (true);

-- hmm_results: inserción pública (para scripts de análisis HMM)
create policy "hmm_results_insert_public"
    on hmm_results for insert
    with check (true);

create policy "hmm_results_update_public"
    on hmm_results for update
    using (true) with check (true);

create policy "hmm_market_results_select_public"
    on hmm_market_results for select using (true);

create policy "hmm_market_results_insert_public"
    on hmm_market_results for insert with check (true);

create policy "hmm_market_results_update_public"
    on hmm_market_results for update
    using (true) with check (true);

-- users: cada usuario solo ve y edita su propio perfil
create policy "users_select_own"
    on users for select
    using (auth.uid() = id);

create policy "users_update_own"
    on users for update
    using (auth.uid() = id);

-- custom_requests: solo el dueño puede ver sus solicitudes
create policy "cr_select_own"
    on custom_requests for select
    using (auth.uid() = user_id);

-- custom_requests: solo usuarios premium pueden insertar
create policy "cr_insert_premium"
    on custom_requests for insert
    with check (
        auth.uid() = user_id
        and exists (
            select 1 from users u
            where u.id = auth.uid()
              and u.plan in ('premium', 'admin')
              and (u.plan_expires_at is null or u.plan_expires_at > now())
        )
    );


-- ============================================================
-- FUNCIÓN DE AUDITORÍA (opcional pero recomendada)
-- Actualiza automáticamente un campo updated_at si lo añades.
-- ============================================================
create or replace function set_updated_at()
returns trigger language plpgsql as $$
begin
    new.updated_at = now();
    return new;
end;
$$;


-- ============================================================
-- DATOS DE EJEMPLO para arrancar
-- ============================================================
insert into stocks (ticker, name, exchange, sector, industry) values
    ('AAPL',  'Apple Inc.',             'NASDAQ', 'Technology',        'Consumer Electronics'),
    ('MSFT',  'Microsoft Corporation',  'NASDAQ', 'Technology',        'Software—Infrastructure'),
    ('GOOGL', 'Alphabet Inc.',          'NASDAQ', 'Technology',        'Internet Content & Info'),
    ('AMZN',  'Amazon.com Inc.',        'NASDAQ', 'Consumer Cyclical', 'Internet Retail'),
    ('NVDA',  'NVIDIA Corporation',     'NASDAQ', 'Technology',        'Semiconductors')
on conflict (ticker) do nothing;
