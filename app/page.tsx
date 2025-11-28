'use client';

import { useState, useCallback } from 'react';
import { Header } from '@/components/Header';
import { Sidebar } from '@/components/Sidebar';
import { MapArea } from '@/components/MapArea';
import { WorkflowSelector } from '@/components/WorkflowSelector';
import { Toaster } from '@/components/ui/sonner';

export interface SegmentationResult {
  overlayData: string; // Base64 encoded image or data URL
  statistics: {
    [key: string]: number;
  };
}

type WorkflowMode = 'dataset' | 'segmentation' | null;

export default function Page() {
  const [workflowMode, setWorkflowMode] = useState<WorkflowMode>(null);
  const [segmentationResult, setSegmentationResult] = useState<SegmentationResult | null>(null);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(80);
  const [showBox, setShowBox] = useState(true);
  const [mapCenter, setMapCenter] = useState({ lat: 36.5674, lng: -121.9500 }); // Pebble Beach
  const [capturedImageData, setCapturedImageData] = useState<string | undefined>();
  const [triggerCapture, setTriggerCapture] = useState(false);

  // Memoized callbacks to prevent infinite re-renders
  const handleSegmentationComplete = useCallback((result: SegmentationResult) => {
    setSegmentationResult(result);
    setIsSegmenting(false);
  }, []);

  const handleImageCapture = useCallback((imageData: string) => {
    setCapturedImageData(imageData);
  }, []);

  // Show workflow selector if no mode is selected
  if (!workflowMode) {
    return (
      <div className="h-screen flex flex-col bg-slate-950">
        <Header onLogoClick={() => setWorkflowMode(null)} />
        <WorkflowSelector onSelectWorkflow={setWorkflowMode} />
        <Toaster />
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-slate-950">
      <Header onLogoClick={() => setWorkflowMode(null)} />
      
      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          workflowMode={workflowMode}
          segmentationResult={segmentationResult}
          isSegmenting={isSegmenting}
          overlayOpacity={overlayOpacity}
          showBox={showBox}
          mapCenter={mapCenter}
          capturedImageData={capturedImageData}
          onSegment={() => setIsSegmenting(true)}
          onClear={() => setSegmentationResult(null)}
          onOpacityChange={setOverlayOpacity}
          onShowBoxChange={setShowBox}
          onLocationSelect={setMapCenter}
          onSwitchWorkflow={(mode) => {
            setWorkflowMode(mode);
            setSegmentationResult(null);
          }}
          onTriggerCapture={() => setTriggerCapture(true)}
        />
        
        <MapArea
          center={mapCenter}
          segmentationResult={segmentationResult}
          isSegmenting={isSegmenting}
          overlayOpacity={overlayOpacity}
          showBox={showBox}
          triggerCapture={triggerCapture}
          onCaptureComplete={() => setTriggerCapture(false)}
          onSegmentationComplete={handleSegmentationComplete}
          onImageCapture={handleImageCapture}
        />
      </div>
      
      <Toaster />
    </div>
  );
}
