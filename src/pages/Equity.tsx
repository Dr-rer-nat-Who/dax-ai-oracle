
import { Card } from "@/components/ui/card";
// import { LineChart, Line, XAxis, YAxis, Tooltip } from "recharts"; // For real data

export default function Equity() {
  // Normally, chart data would be fetched - we use dummy data for now:
  const dummy = [
    { date: "09:31", value: 100000 },
    { date: "09:32", value: 100320 },
    { date: "09:33", value: 100180 },
    { date: "09:34", value: 100670 },
    { date: "09:35", value: 101140 },
    { date: "09:36", value: 101511 },
  ];

  return (
    <div className="flex flex-col items-center max-w-5xl mx-auto pt-10 animate-fade-in">
      <Card className="p-6 w-full shadow-md flex flex-col gap-4">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-xl font-semibold">Equity Chart</h1>
          <span className="text-xs text-muted-foreground">Demo: Vectorbt Equity Curve</span>
        </div>
        <div className="bg-background rounded-lg w-full p-4 border border-border flex justify-center items-end min-h-[250px]">
          {/* Platzhalter-Chart (recharts Integration später) */}
          <svg viewBox="0 0 430 140" width="100%" height="140">
            <polyline
              fill="none"
              stroke="#2563eb"
              strokeWidth="3"
              points="0,120 72,110 144,115 216,95 288,60 360,40 430,70"
              style={{ filter: "drop-shadow(0 1px 2px #60a5fa33)" }}
            />
            <circle cx="360" cy="40" r="3" fill="#22d3ee" />
            <circle cx="430" cy="70" r="3" fill="#22d3ee" />
          </svg>
        </div>
        <div className="mt-2 text-sm text-muted-foreground">Equity-Daten werden im Livemodus automatisch aktualisiert.</div>
      </Card>
    </div>
  );
}
