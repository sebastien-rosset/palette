import LoginClient from './LoginClient'
import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'

export default async function LoginPage() {
  const supabase = await createClient()
  
  // Optional: Check if user is already logged in
  const { data: { session } } = await supabase.auth.getSession()
  if (session) {
    redirect('/dashboard') // Redirect to dashboard if already logged in
  }

  return <LoginClient />
}