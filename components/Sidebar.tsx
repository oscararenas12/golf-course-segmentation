'use client';

import { Search, MapPin, Target, Trash2, ArrowLeftRight } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Checkbox } from './ui/checkbox';
import { Legend } from './Legend';
import { Statistics } from './Statistics';
import { DatasetCapture } from './DatasetCapture';
import { SegmentationResult } from '../app/page';
import { Separator } from './ui/separator';

type WorkflowMode = 'dataset' | 'segmentation';

interface SidebarProps {
  workflowMode: WorkflowMode;
  segmentationResult: SegmentationResult | null;
  isSegmenting: boolean;
  overlayOpacity: number;
  showBox: boolean;
  mapCenter: { lat: number; lng: number };
  capturedImageData?: string;
  onSegment: () => void;
  onClear: () => void;
  onOpacityChange: (value: number) => void;
  onShowBoxChange: (value: boolean) => void;
  onLocationSelect: (location: { lat: number; lng: number }) => void;
  onSwitchWorkflow: (mode: WorkflowMode) => void;
  onTriggerCapture: () => void;
}

const FAMOUS_COURSES = [
  { name: 'Pebble Beach Golf Links', lat: 36.5674, lng: -121.9500 },
  { name: 'Augusta National Golf Club', lat: 33.5027, lng: -82.0199 },
  { name: 'St Andrews Old Course', lat: 56.3459, lng: -2.8238 },
  { name: 'Pinehurst No. 2', lat: 35.1896, lng: -79.4689 },
  { name: 'Oakmont Country Club', lat: 40.5229, lng: -79.8542 },
];

export function Sidebar({
  workflowMode,
  segmentationResult,
  isSegmenting,
  overlayOpacity,
  showBox,
  mapCenter,
  capturedImageData,
  onSegment,
  onClear,
  onOpacityChange,
  onShowBoxChange,
  onLocationSelect,
  onSwitchWorkflow,
  onTriggerCapture,
}: SidebarProps) {
  return (
    <aside className="w-[300px] bg-slate-800 border-r border-slate-700 overflow-y-auto">
      <div className="p-6 space-y-6">
        {/* Workflow Mode Indicator */}
        <div className="bg-slate-700/30 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-slate-400 text-xs uppercase tracking-wide">Current Mode</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onSwitchWorkflow(workflowMode === 'dataset' ? 'segmentation' : 'dataset')}
              className="h-6 px-2 text-slate-400 hover:text-slate-200"
            >
              <ArrowLeftRight className="size-3" />
            </Button>
          </div>
          <div className="flex items-center gap-2">
            {workflowMode === 'dataset' ? (
              <>
                <div className="size-2 bg-blue-400 rounded-full" />
                <span className="text-slate-200">Dataset Creation</span>
              </>
            ) : (
              <>
                <div className="size-2 bg-emerald-400 rounded-full" />
                <span className="text-slate-200">Segmentation Analysis</span>
              </>
            )}
          </div>
        </div>

        <Separator className="bg-slate-700" />

        {/* Search Section */}
        <div>
          <label className="text-slate-300 flex items-center gap-2 mb-2">
            <Search className="size-4" />
            Search Location
          </label>
          <select
            className="w-full bg-slate-700 text-slate-200 border border-slate-600 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-emerald-500"
            onChange={(e) => {
              const course = FAMOUS_COURSES[parseInt(e.target.value)];
              if (course) {
                onLocationSelect({ lat: course.lat, lng: course.lng });
              }
            }}
          >
            <option value="">Search for a golf course...</option>
            {FAMOUS_COURSES.map((course, index) => (
              <option key={index} value={index}>
                {course.name}
              </option>
            ))}
          </select>
        </div>

        <Separator className="bg-slate-700" />

        {/* Instructions */}
        <div className="bg-slate-700/50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <MapPin className="size-4 text-emerald-400" />
            <h3 className="text-slate-200">Position Box</h3>
          </div>
          <p className="text-slate-400 text-sm">
            {workflowMode === 'dataset' 
              ? 'Pan & zoom the map to frame your golf course, then capture the image'
              : 'Pan & zoom the map to frame your golf course within the capture box'
            }
          </p>
        </div>

        {/* Segment Button - Only show in segmentation mode */}
        {workflowMode === 'segmentation' && (
          <>
            <Button
              onClick={onSegment}
              disabled={isSegmenting || !!segmentationResult}
              className="w-full h-12 bg-emerald-500 hover:bg-emerald-600 text-white disabled:bg-slate-600 disabled:text-slate-400"
              size="lg"
            >
              {isSegmenting ? (
                <>
                  <div className="size-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Target className="size-4 mr-2" />
                  Segment Course
                </>
              )}
            </Button>
            <Separator className="bg-slate-700" />
          </>
        )}

        {/* Dataset Capture Section - Only show in dataset mode */}
        {workflowMode === 'dataset' && (
          <>
            <DatasetCapture
              mapCenter={mapCenter}
              segmentationResult={segmentationResult}
              capturedImageData={capturedImageData}
              workflowMode={workflowMode}
              onTriggerCapture={onTriggerCapture}
            />
            <Separator className="bg-slate-700" />
          </>
        )}

        {/* Legend - Only show in segmentation mode */}
        {workflowMode === 'segmentation' && (
          <>
            <Legend />
            <Separator className="bg-slate-700" />
          </>
        )}

        {/* Statistics (if segmented) */}
        {segmentationResult && (
          <>
            <Statistics statistics={segmentationResult.statistics} />
            <Separator className="bg-slate-700" />
          </>
        )}

        {/* Settings - Only show in segmentation mode */}
        {workflowMode === 'segmentation' && (
          <div>
            <h3 className="text-slate-200 flex items-center gap-2 mb-4">
              ⚙️ Settings
            </h3>
            
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-2">
                  <label className="text-slate-300 text-sm">Overlay Opacity</label>
                  <span className="text-slate-400 text-sm">{overlayOpacity}%</span>
                </div>
                <input
                  type="range"
                  value={overlayOpacity}
                  onChange={(e) => onOpacityChange(Number(e.target.value))}
                  min={0}
                  max={100}
                  step={5}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
              </div>

              <div className="flex items-center gap-2">
                <Checkbox
                  id="showBox"
                  checked={showBox}
                  onCheckedChange={(checked) => onShowBoxChange(checked as boolean)}
                />
                <label htmlFor="showBox" className="text-slate-300 text-sm cursor-pointer">
                  Show Capture Box
                </label>
              </div>

              {segmentationResult && (
                <Button
                  onClick={onClear}
                  variant="outline"
                  className="w-full border-slate-600 text-slate-300 hover:bg-slate-700"
                >
                  <Trash2 className="size-4 mr-2" />
                  Clear Segmentation
                </Button>
              )}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}