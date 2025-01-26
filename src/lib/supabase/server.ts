import { createClient as createSupabaseClient } from '@supabase/supabase-js'
import { cookies } from 'next/headers'
import type { CookieOptions } from '@supabase/ssr'

export const createClient = async () => {
  const cookieStore = cookies()

  return createSupabaseClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookieOptions: {
        name: 'sb-access-token',
        domain: process.env.NODE_ENV === 'production' 
          ? process.env.NEXT_PUBLIC_SITE_URL?.replace('https://', '') 
          : undefined,
        path: '/',
        secure: process.env.NODE_ENV === 'production',
      },
      cookies: {
        get(name: string) {
          return cookieStore.get(name)?.value ?? null
        },
        set(name: string, value: string, options: CookieOptions) {
          try {
            cookieStore.set({ 
              name, 
              value, 
              ...options 
            })
          } catch (error) {
            console.error('Error setting cookie:', error)
          }
        },
        remove(name: string, options: CookieOptions) {
          try {
            cookieStore.delete({ 
              name, 
              ...options 
            })
          } catch (error) {
            console.error('Error removing cookie:', error)
          }
        },
      },
      cookieEncoding: 'raw'
    }
  )
}