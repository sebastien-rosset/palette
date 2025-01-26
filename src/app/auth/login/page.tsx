import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'

export default async function LoginPage() {
  const supabase = createClient()

  const signInWithGoogle = async () => {
    'use server'
    const { data, error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: `${process.env.NEXT_PUBLIC_SITE_URL}/auth/callback`,
      },
    })
    if (error) {
      return redirect('/auth/error')
    }
    return redirect(data.url)
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

        <div className="mt-8 space-y-4">
          <form action={signInWithGoogle}>
            <button
              type="submit"
              className="w-full rounded-md bg-white px-4 py-2 text-base font-medium text-gray-700 shadow-sm ring-1 ring-gray-300 hover:bg-gray-50"
            >
              Continue with Google
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}