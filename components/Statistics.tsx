import { BarChart3 } from 'lucide-react';

interface ClassStats {
  pixels: number;
  percentage: number;
}

interface StatisticsProps {
  statistics: {
    [key: string]: number | ClassStats;
  };
}

export function Statistics({ statistics }: StatisticsProps) {
  // Handle both formats: { class: number } and { class: { pixels, percentage } }
  const getPercentage = (value: number | ClassStats): number => {
    if (typeof value === 'number') return value;
    return value.percentage;
  };

  return (
    <div>
      <h3 className="text-slate-200 flex items-center gap-2 mb-3">
        <BarChart3 className="size-4" />
        Results
      </h3>
      <div className="space-y-2">
        {Object.entries(statistics)
          .sort((a, b) => getPercentage(b[1]) - getPercentage(a[1]))
          .map(([name, value]) => (
            <div key={name} className="flex justify-between items-center">
              <span className="text-slate-300 text-sm">{name}:</span>
              <span className="text-slate-400 text-sm">{getPercentage(value).toFixed(1)}%</span>
            </div>
          ))}
      </div>
    </div>
  );
}
