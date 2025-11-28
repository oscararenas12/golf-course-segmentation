const LEGEND_ITEMS = [
  { name: 'Background', color: '#1a1a1a' },
  { name: 'Fairway', color: '#2d5016' },
  { name: 'Green', color: '#4ade80' },
  { name: 'Tee', color: '#ef4444' },
  { name: 'Bunker', color: '#fbbf24' },
  { name: 'Water', color: '#3b82f6' },
];

export function Legend() {
  return (
    <div>
      <h3 className="text-slate-200 flex items-center gap-2 mb-3">
        🎨 Legend
      </h3>
      <div className="space-y-2">
        {LEGEND_ITEMS.map((item) => (
          <div key={item.name} className="flex items-center gap-2">
            <div
              className="size-4 rounded-full border border-slate-600"
              style={{ backgroundColor: item.color }}
            />
            <span className="text-slate-300 text-sm">{item.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
