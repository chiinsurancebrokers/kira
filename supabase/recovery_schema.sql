-- Asklepios · Recovery at home (post-discharge monitoring)
-- Same pattern as public.drafts: the app writes Fernet-encrypted JSON in `data`,
-- RLS is ON, and one permissive policy lets the app's anon key read/write
-- (the app filters by user_email itself).  Review before running.

create table if not exists public.recovery_plans (
  user_email text not null,
  plan_id    text not null,
  data       text not null,                       -- encrypted plan JSON
  updated_at timestamptz not null default now(),
  primary key (user_email, plan_id)
);

create table if not exists public.recovery_entries (
  user_email text not null,
  plan_id    text not null,
  entry_id   text not null,
  ts         timestamptz not null,
  data       text not null,                       -- encrypted check-in JSON
  primary key (user_email, plan_id, entry_id)
);
create index if not exists recovery_entries_plan_ts on public.recovery_entries (user_email, plan_id, ts desc);

alter table public.recovery_plans   enable row level security;
alter table public.recovery_entries enable row level security;

create policy "app access recovery_plans"   on public.recovery_plans   for all to anon, authenticated using (true) with check (true);
create policy "app access recovery_entries" on public.recovery_entries for all to anon, authenticated using (true) with check (true);
