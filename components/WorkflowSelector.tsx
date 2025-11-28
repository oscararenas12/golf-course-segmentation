'use client';

import { Database, Target, ArrowRight, Image, BarChart3 } from 'lucide-react';
import { Button } from './ui/button';

interface WorkflowSelectorProps {
  onSelectWorkflow: (workflow: 'dataset' | 'segmentation') => void;
}

export function WorkflowSelector({ onSelectWorkflow }: WorkflowSelectorProps) {
  return (
    <div className="flex-1 flex items-center justify-center bg-slate-950">
      <div className="max-w-5xl w-full px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <span className="text-6xl">⛳</span>
          </div>
          <h1 className="text-white text-4xl mb-3">
            Golf Course Segmentation
          </h1>
          <p className="text-slate-400 text-lg">
            Choose your workflow to get started
          </p>
        </div>

        {/* Workflow Options */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Dataset Creation Workflow */}
          <button
            onClick={() => onSelectWorkflow('dataset')}
            className="group relative bg-slate-800 border-2 border-slate-700 rounded-2xl p-8 text-left transition-all hover:border-blue-500 hover:shadow-xl hover:shadow-blue-500/20 hover:-translate-y-1"
          >
            {/* Best For Badge */}
            <div className="absolute top-4 left-4 px-3 py-1 bg-blue-500/20 text-blue-400 text-xs rounded-full border border-blue-500/30">
              Best for: ML Training
            </div>

            <div className="absolute top-4 right-4 size-12 bg-blue-500/10 rounded-full flex items-center justify-center group-hover:bg-blue-500/20 transition-colors">
              <Database className="size-6 text-blue-400" />
            </div>

            <div className="mb-6 mt-8">
              <h2 className="text-white text-2xl mb-2">
                Create a Dataset
              </h2>
              <p className="text-slate-400">
                Build a custom dataset of golf course imagery for training ML models
              </p>
            </div>

            <div className="space-y-3 mb-6">
              <div className="flex items-start gap-3 text-slate-300 text-sm">
                <div className="size-5 bg-blue-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="size-2 bg-blue-400 rounded-full" />
                </div>
                <span>Capture multiple golf course locations</span>
              </div>
              <div className="flex items-start gap-3 text-slate-300 text-sm">
                <div className="size-5 bg-blue-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="size-2 bg-blue-400 rounded-full" />
                </div>
                <span>Save satellite imagery + segmentation masks</span>
              </div>
              <div className="flex items-start gap-3 text-slate-300 text-sm">
                <div className="size-5 bg-blue-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="size-2 bg-blue-400 rounded-full" />
                </div>
                <span>Export as training-ready dataset</span>
              </div>
            </div>

            <div className="flex items-center gap-2 text-blue-400 group-hover:gap-3 transition-all">
              <span>Start Building</span>
              <ArrowRight className="size-4" />
            </div>
          </button>

          {/* Segmentation Workflow */}
          <button
            onClick={() => onSelectWorkflow('segmentation')}
            className="group relative bg-slate-800 border-2 border-slate-700 rounded-2xl p-8 text-left transition-all hover:border-emerald-500 hover:shadow-xl hover:shadow-emerald-500/20 hover:-translate-y-1"
          >
            {/* Best For Badge */}
            <div className="absolute top-4 left-4 px-3 py-1 bg-emerald-500/20 text-emerald-400 text-xs rounded-full border border-emerald-500/30">
              Best for: Analysis
            </div>

            <div className="absolute top-4 right-4 size-12 bg-emerald-500/10 rounded-full flex items-center justify-center group-hover:bg-emerald-500/20 transition-colors">
              <Target className="size-6 text-emerald-400" />
            </div>

            <div className="mb-6 mt-8">
              <h2 className="text-white text-2xl mb-2">
                Run Segmentation
              </h2>
              <p className="text-slate-400">
                Analyze individual golf courses to identify features and areas
              </p>
            </div>

            <div className="space-y-3 mb-6">
              <div className="flex items-start gap-3 text-slate-300 text-sm">
                <div className="size-5 bg-emerald-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="size-2 bg-emerald-400 rounded-full" />
                </div>
                <span>Search and select any golf course</span>
              </div>
              <div className="flex items-start gap-3 text-slate-300 text-sm">
                <div className="size-5 bg-emerald-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="size-2 bg-emerald-400 rounded-full" />
                </div>
                <span>AI-powered feature detection</span>
              </div>
              <div className="flex items-start gap-3 text-slate-300 text-sm">
                <div className="size-5 bg-emerald-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="size-2 bg-emerald-400 rounded-full" />
                </div>
                <span>View statistics and overlay visualization</span>
              </div>
            </div>

            <div className="flex items-center gap-2 text-emerald-400 group-hover:gap-3 transition-all">
              <span>Start Analyzing</span>
              <ArrowRight className="size-4" />
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}