-- Allow one generation job to produce multiple clip rows.
alter table public.clips drop constraint if exists clips_job_id_key;
create index if not exists clips_job_id_idx on public.clips(job_id);
