import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export default function Home() {
  return (
    <div className="container mx-auto py-8">
      <h1 className="text-4xl font-bold mb-8">Art Inventory</h1>
      
      {/* Grid of artwork cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Example artwork card */}
        <Card>
          <CardHeader>
            <CardTitle>Sample Artwork</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="aspect-square bg-gray-200 mb-4 rounded-lg"></div>
            <div className="space-y-2">
              <p><span className="font-semibold">Date:</span> 2024</p>
              <p><span className="font-semibold">Medium:</span> Oil on canvas</p>
              <p><span className="font-semibold">Dimensions:</span> 60 x 80 cm</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}