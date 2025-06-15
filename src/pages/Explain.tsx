
import { Card } from "@/components/ui/card";

const features = [
  { name: "ATR_14", importance: 0.27 },
  { name: "WickRatio", importance: 0.19 },
  { name: "DayOfWeek", importance: 0.09 },
  { name: "RSI", importance: 0.07 },
  { name: "Window64", importance: 0.04 },
];

export default function Explain() {
  return (
    <div className="flex flex-col items-center max-w-4xl mx-auto pt-10 animate-fade-in">
      <Card className="p-6 w-full shadow-md flex flex-col gap-6">
        <h1 className="text-xl font-semibold">Feature Importance (LightGBM)</h1>
        <ul className="space-y-2">
          {features.map(f => (
            <li key={f.name} className="flex items-center gap-4">
              <span className="w-36 font-mono text-xs text-muted-foreground">{f.name}</span>
              <div className="flex-1 bg-muted h-4 rounded overflow-hidden">
                <div
                  className="bg-primary h-4 rounded-l"
                  style={{ width: `${100 * f.importance}%`, minWidth: 4 }}
                />
              </div>
              <span className="w-10 text-right text-xs font-semibold tabular-nums">{(f.importance * 100).toFixed(0)}%</span>
            </li>
          ))}
        </ul>
        <div className="mt-8">
          <h2 className="text-lg font-semibold mb-2 text-muted-foreground">TFT Attention-Heatmap</h2>
          <div className="rounded-lg bg-muted min-h-[60px] flex items-center justify-center text-muted-foreground animate-fade-in">[Heatmap Platzhalter]</div>
        </div>
      </Card>
    </div>
  );
}
