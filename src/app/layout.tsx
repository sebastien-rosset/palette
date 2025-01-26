import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Palette | Your Digital Art Collection',
  description: 'A modern digital workspace for managing your art collection',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <header className="border-b">
          <div className="container mx-auto px-4 py-4">
            <h1 className="text-2xl font-bold text-gray-900">Palette</h1>
          </div>
        </header>
        <main className="min-h-screen bg-gray-50">
          {children}
        </main>
        <footer className="border-t">
          <div className="container mx-auto px-4 py-4 text-sm text-gray-600">
            © {new Date().getFullYear()} Palette. All rights reserved.
          </div>
        </footer>
      </body>
    </html>
  )
}