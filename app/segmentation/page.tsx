'use client';

import { useState, useCallback } from 'react';
import Link from 'next/link';
import { ArrowLeft, MapPin, Scan, Sliders } from 'lucide-react';
import { Header } from '@/components/Header';
import { Footer } from '@/components/Footer';
import { MapArea } from '@/components/MapArea';
import { Statistics } from '@/components/Statistics';
import { Legend } from '@/components/Legend';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Slider } from '@/components/ui/slider';
import { Toaster } from '@/components/ui/sonner';
import { SegmentationResult } from '@/types/segmentation';

const FAMOUS_COURSES = [
  { name: 'Pebble Beach Golf Links', lat: 36.5674, lng: -121.9500 },
  { name: 'Augusta National Golf Club', lat: 33.5027, lng: -82.0199 },
  { name: 'St Andrews Old Course', lat: 56.3459, lng: -2.8238 },
  { name: 'Pinehurst No. 2', lat: 35.1896, lng: -79.4689 },
  { name: 'Oakmont Country Club', lat: 40.5229, lng: -79.8542 },
];

export default function SegmentationPage() {
  const [mapCenter, setMapCenter] = useState({ lat: 36.5674, lng: -121.9500 });
  const [segmentationResult, setSegmentationResult] = useState<SegmentationResult | null>(null);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(80);
  const [triggerCapture, setTriggerCapture] = useState(false);

  const handleSegmentationComplete = useCallback((result: SegmentationResult) => {
    setSegmentationResult(result);
    setIsSegmenting(false);
  }, []);

  const handleAnalyze = () => {
    setIsSegmenting(true);
    setTriggerCapture(true);
  };

  const handleClearResults = () => {
    setSegmentationResult(null);
  };

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
            <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-3">
              <h2 className="text-emerald-400 font-medium flex items-center gap-2">
                <span className="text-lg">🎯</span>
                Segmentation Analysis
              </h2>
              <p className="text-slate-400 text-sm mt-1">
                AI-powered golf course feature detection
              </p>
            </div>

            <Separator className="bg-slate-700" />

            {/* Location */}
            <div>
              <h3 className="text-slate-200 flex items-center gap-2 mb-3">
                <MapPin className="size-4" />
                Location
              </h3>
              <div className="space-y-1">
                {FAMOUS_COURSES.map((course) => (
                  <button
                    key={course.name}
                    onClick={() => {
                      setMapCenter({ lat: course.lat, lng: course.lng });
                      setSegmentationResult(null);
                    }}
                    className="w-full text-left px-3 py-2 rounded text-sm text-slate-300 hover:bg-slate-700 flex items-center gap-2"
                  >
                    <MapPin className="size-3 text-slate-500" />
                    {course.name}
                  </button>
                ))}
              </div>
            </div>

            <Separator className="bg-slate-700" />

            {/* Analyze Controls */}
            <div>
              <h3 className="text-slate-200 flex items-center gap-2 mb-3">
                <Scan className="size-4" />
                Analysis
              </h3>
              <Button
                onClick={handleAnalyze}
                disabled={isSegmenting}
                className="w-full bg-emerald-600 hover:bg-emerald-700 mb-3"
              >
                {isSegmenting ? (
                  <>
                    <div className="size-4 mr-2 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Scan className="size-4 mr-2" />
                    Analyze Golf Course
                  </>
                )}
              </Button>

              {segmentationResult && (
                <Button
                  onClick={handleClearResults}
                  className="w-full bg-slate-600 text-white hover:bg-slate-500"
                >
                  Clear Results
                </Button>
              )}
            </div>

            {/* Overlay Controls */}
            {segmentationResult && (
              <>
                <Separator className="bg-slate-700" />
                <div>
                  <h3 className="text-slate-200 flex items-center gap-2 mb-3">
                    <Sliders className="size-4" />
                    Overlay
                  </h3>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">Opacity</span>
                      <span className="text-slate-300">{overlayOpacity}%</span>
                    </div>
                    <Slider
                      value={[overlayOpacity]}
                      onValueChange={([value]) => setOverlayOpacity(value)}
                      min={0}
                      max={100}
                      step={5}
                      className="w-full"
                    />
                  </div>
                </div>
              </>
            )}

            <Separator className="bg-slate-700" />

            {/* Legend */}
            <Legend />

            {/* Statistics */}
            {segmentationResult && (
              <>
                <Separator className="bg-slate-700" />
                <Statistics statistics={segmentationResult.statistics} />
              </>
            )}
          </div>
        </aside>

        {/* Map Area */}
        <MapArea
          center={mapCenter}
          segmentationResult={segmentationResult}
          isSegmenting={isSegmenting}
          overlayOpacity={overlayOpacity}
          showBox={true}
          triggerCapture={triggerCapture}
          onCaptureComplete={() => setTriggerCapture(false)}
          onSegmentationComplete={handleSegmentationComplete}
          onImageCapture={() => {}}
        />
      </div>

      <Footer />
      <Toaster />
    </div>
  );
}
