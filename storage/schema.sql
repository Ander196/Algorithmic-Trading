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

    -- Regime detection fields
    regime_label            varchar(20)     not null,
    state_id                integer         not null,
    probability             numeric(6, 4)   not null,
    state_probabilities     jsonb,
    volatility_level        varchar(20),

    -- Multiplier breakdown
    final_multiplier        numeric(4, 2)   not null,
    base_multiplier         numeric(4, 2),
    vol_scalar              numeric(4, 2),

    -- Stock-specific fields
    beta                    numeric(6, 4),
    relative_vol            numeric(6, 4),

    -- Regime state flags
    is_regime_confirmed     boolean         not null default false,
    is_flickering           boolean         not null default false,
    stability_bars          integer,

    -- Additional metadata
    reasoning               text,
    created_at               timestamptz     not null default now(),

    constraint uq_hmm_results_two_layer_ticker_date unique (ticker, result_date)
);

CREATE INDEX IF NOT EXISTS idx_hmm_results_two_layer_ticker_date
    ON hmm_results (ticker, result_date desc);

CREATE INDEX IF NOT EXISTS idx_hmm_results_two_layer_date
    ON hmm_results (result_date desc);

-- RLS policies (same as original hmm_results)
ALTER TABLE hmm_results enable row level security;

CREATE POLICY "hmm_results_two_layer_select_public"
    ON hmm_results for select using (true);

CREATE POLICY "hmm_results_two_layer_insert_public"
    ON hmm_results for insert with check (true);


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


-- ============================================================
-- ROW LEVEL SECURITY (RLS)
-- Habilitar en todas las tablas de negocio.
-- ============================================================

alter table stocks          enable row level security;
alter table stock_prices    enable row level security;
alter table signals         enable row level security;
alter table hmm_results    enable row level security;
alter table users           enable row level security;
alter table custom_requests enable row level security;

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
