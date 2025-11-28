import { BarChart3 } from 'lucide-react';

interface StatisticsProps {
  statistics: {
    [key: string]: number;
  };
}

export function Statistics({ statistics }: StatisticsProps) {
  return (
    <div>
      <h3 className="text-slate-200 flex items-center gap-2 mb-3">
        <BarChart3 className="size-4" />
        Results
      </h3>
      <div className="space-y-2">
        {Object.entries(statistics)
          .sort((a, b) => b[1] - a[1])
          .map(([name, percentage]) => (
            <div key={name} className="flex justify-between items-center">
              <span className="text-slate-300 text-sm">{name}:</span>
              <span className="text-slate-400 text-sm">{percentage.toFixed(1)}%</span>
            </div>
          ))}
      </div>
    </div>
  );
}
