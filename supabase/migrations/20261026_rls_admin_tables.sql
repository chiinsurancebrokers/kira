-- Close the four tables that had RLS off (anyone with the public anon key could read/write them).
-- Reads of active rows stay public (the app shows banners/articles/partners); all writes and the
-- login log go through the server-side service-role client (SUPABASE_SERVICE_ROLE_KEY).
alter table public.partners   enable row level security;
alter table public.articles   enable row level security;
alter table public.campaigns  enable row level security;
alter table public.user_logins enable row level security;

drop policy if exists "public read active" on public.partners;
drop policy if exists "public read active" on public.articles;
drop policy if exists "public read active" on public.campaigns;
create policy "public read active" on public.partners  for select to anon, authenticated using (coalesce(active, true));
create policy "public read active" on public.articles  for select to anon, authenticated using (coalesce(active, true));
create policy "public read active" on public.campaigns for select to anon, authenticated using (coalesce(active, true));
-- user_logins: no policy = no access for anon/authenticated (service role bypasses RLS).
