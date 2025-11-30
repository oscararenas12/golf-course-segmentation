'use client';

import { useState, useCallback } from 'react';
import Link from 'next/link';
import { ArrowLeft, Search, MapPin } from 'lucide-react';
import { Header } from '@/components/Header';
import { Footer } from '@/components/Footer';
import { MapArea } from '@/components/MapArea';
import { ClassificationCapture } from '@/components/ClassificationCapture';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Separator } from '@/components/ui/separator';
import { Toaster } from '@/components/ui/sonner';
import { SegmentationResult } from '@/types/segmentation';

const SAMPLE_LOCATIONS = [
  { name: 'Pebble Beach Golf Links', lat: 36.5674, lng: -121.9500 },
  { name: 'Central Park NYC', lat: 40.7829, lng: -73.9654 },
  { name: 'Augusta National', lat: 33.5027, lng: -82.0199 },
  { name: 'Farmland Iowa', lat: 41.8780, lng: -93.0977 },
  { name: 'LAX Airport', lat: 33.9416, lng: -118.4085 },
];

export default function ClassificationPage() {
  const [mapCenter, setMapCenter] = useState({ lat: 36.5674, lng: -121.9500 });
  const [capturedImageData, setCapturedImageData] = useState<string | undefined>();
  const [triggerCapture, setTriggerCapture] = useState(false);

  const handleImageCapture = useCallback((imageData: string) => {
    setCapturedImageData(imageData);
  }, []);

  // Dummy handler - classification doesn't use segmentation
  const handleSegmentationComplete = useCallback((result: SegmentationResult) => {
    // Not used in classification mode
  }, []);

  return (
    <div className="h-screen flex flex-col bg-slate-950">
      <Header />

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-[300px] bg-slate-800 border-r border-slate-700 overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Back Link */}
            <Link href="/">
              <Button className="bg-slate-600 text-white hover:bg-slate-500 -ml-2">
                <ArrowLeft className="size-4 mr-2" />
                Back to Home
              </Button>
            </Link>

            {/* Page Title */}
            <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-3">
              <h2 className="text-orange-400 font-medium flex items-center gap-2">
                <span className="text-lg">🏷️</span>
                Classification Dataset
              </h2>
              <p className="text-slate-400 text-sm mt-1">
                Label images: Golf or Not Golf
              </p>
            </div>

            <Separator className="bg-slate-700" />

            {/* Location Search */}
            <div>
              <h3 className="text-slate-200 flex items-center gap-2 mb-3">
                <Search className="size-4" />
                Sample Locations
              </h3>
              <p className="text-slate-500 text-xs mb-2">
                Mix of golf and non-golf areas
              </p>
              <div className="space-y-1">
                {SAMPLE_LOCATIONS.map((location) => (
                  <button
                    key={location.name}
                    onClick={() => setMapCenter({ lat: location.lat, lng: location.lng })}
                    className="w-full text-left px-3 py-2 rounded text-sm text-slate-300 hover:bg-slate-700 flex items-center gap-2"
                  >
                    <MapPin className="size-3 text-slate-500" />
                    {location.name}
                  </button>
                ))}
              </div>
            </div>

            <Separator className="bg-slate-700" />

            {/* Classification Capture */}
            <ClassificationCapture
              mapCenter={mapCenter}
              capturedImageData={capturedImageData}
              onTriggerCapture={() => setTriggerCapture(true)}
            />
          </div>
        </aside>

        {/* Map Area */}
        <MapArea
          center={mapCenter}
          segmentationResult={null}
          isSegmenting={false}
          overlayOpacity={80}
          showBox={true}
          triggerCapture={triggerCapture}
          onCaptureComplete={() => setTriggerCapture(false)}
          onSegmentationComplete={handleSegmentationComplete}
          onImageCapture={handleImageCapture}
        />
      </div>

      <Footer />
      <Toaster />
    </div>
  );
}
