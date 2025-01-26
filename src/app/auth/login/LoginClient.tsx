'use client'

import { createClientSupabaseClient } from '@/lib/supabase/client'
import { useState } from 'react'
import { useRouter } from 'next/navigation'

export default function LoginClient() {
  const supabase = createClientSupabaseClient()
  const router = useRouter()
  const [error, setError] = useState<string | null>(null)

  const signInWithGoogle = async () => {
    const { data, error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: `${process.env.NEXT_PUBLIC_SITE_URL}/auth/callback`,
      },
    })

    if (error) {
      setError('Failed to initiate Google Sign-In')
      console.error('Google Sign-In Error:', error)
      return
    }

    if (data.url) {
      window.location.href = data.url
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="w-full max-w-md space-y-8 p-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold">Sign in to Palette</h1>
          <p className="mt-2 text-gray-600">
            Manage your art inventory
          </p>
        </div>

        {error && (
          <div className="text-red-500 text-center">
            {error}
          </div>
        )}

        <div className="mt-8 space-y-4">
          <button
            onClick={signInWithGoogle}
            className="w-full rounded-md bg-white px-4 py-2 text-base font-medium text-gray-700 shadow-sm ring-1 ring-gray-300 hover:bg-gray-50"
          >
            Continue with Google
          </button>
        </div>
      </div>
    </div>
  )
}