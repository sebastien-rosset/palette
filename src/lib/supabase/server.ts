import * as SupabaseClient from '@supabase/supabase-js'
import { cookies } from 'next/headers'
import type { CookieOptions } from '@supabase/ssr'

export const createClient = async () => {
  const cookieStore = await cookies()

  return SupabaseClient.createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      auth: {
        persistSession: true,
        storage: {
          getItem: (key: string) => {
            return cookieStore.get(key)?.value ?? null
          },
          setItem: (key: string, value: string) => {
            try {
              cookieStore.set({ 
                name: key, 
                value: value,
                path: '/',
                secure: process.env.NODE_ENV === 'production'
              })
            } catch (error) {
              console.error('Error setting cookie:', error)
            }
          },
          removeItem: (key: string) => {
            try {
              cookieStore.delete({ 
                name: key,
                path: '/'
              })
            } catch (error) {
              console.error('Error removing cookie:', error)
            }
          }
        }
      }
    }
  )
}