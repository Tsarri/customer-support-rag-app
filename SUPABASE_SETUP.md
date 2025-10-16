# Supabase Setup Guide

## 1. Create Supabase Project

1. Go to https://supabase.com
2. Sign up/login
3. Click "New Project"
4. Name it "kntrkt-support" (or any name)
5. Set a strong database password
6. Select a region close to you
7. Wait ~2 minutes for project creation

## 2. Create Database Tables

1. In your Supabase project, go to SQL Editor
2. Copy the contents of `supabase_schema.sql`
3. Paste and click "Run"
4. You should see: "Success. No rows returned"

## 3. Get Your API Credentials

1. Go to Project Settings → API
2. Copy your:
   - Project URL (looks like: https://xxxxx.supabase.co)
   - anon/public key (long string starting with "eyJ...")

## 4. Configure Backend

Add to your environment variables:

```bash
export SUPABASE_URL="https://xxxxx.supabase.co"
export SUPABASE_KEY="eyJ..."
```

Or create a `.env` file:

```
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=eyJ...
```

## 5. Install Supabase Client

```bash
pip3 install --break-system-packages supabase
```

## 6. Restart Backend

```bash
python3 customer_support_agent.py
```

You should see: "✅ Supabase connected"

## Done!

Your admin dashboard will now show real ticket data!
