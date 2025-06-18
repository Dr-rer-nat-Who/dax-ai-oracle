
import { useState } from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

const tickers = [
  { value: "DAX", label: "DAX" },
  { value: "SAP", label: "SAP" },
  { value: "BAS", label: "BASF" },
];

const liveSignals = [
  { ticker: "DAX", signal: "BUY", price: 18780.2, time: "09:34:41" },
  { ticker: "SAP", signal: "SELL", price: 180.7, time: "09:34:41" },
];

export default function Live() {
  const [selected, setSelected] = useState("DAX");

  // In real app: fetch data for ticker
  const ticker = tickers.find(t => t.value === selected);

  return (
    <div className="flex flex-col items-center max-w-4xl mx-auto pt-10 gap-8 animate-fade-in">
      <Card className="p-6 w-full flex flex-col gap-4 shadow-md">
        <div className="flex flex-col sm:flex-row sm:items-center gap-4">
          <Select
            value={selected}
            onValueChange={setSelected}
          >
            <SelectTrigger className="w-40">
              <SelectValue placeholder="Ticker wählen" />
            </SelectTrigger>
            <SelectContent>
              {tickers.map(t => (
                <SelectItem key={t.value} value={t.value}>{t.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <div className="flex-1 flex items-center gap-4 ml-1">
            <span className="text-lg font-semibold text-foreground">{ticker?.label}</span>
            <span className="text-2xl font-bold text-primary">{liveSignals.find(s => s.ticker === selected)?.price ?? <Skeleton className="h-8 w-20" />}</span>
            <span className="text-sm text-muted-foreground">EUR</span>
            <span className="ml-4 text-xs text-muted-foreground">
              {liveSignals.find(s => s.ticker === selected)?.time ?? <Skeleton className="h-4 w-16" />}
            </span>
          </div>
        </div>
        <div>
          <h2 className="text-md font-semibold text-muted-foreground">Aktuelle Signale</h2>
          <div className="flex flex-wrap gap-4 mt-2">
            {liveSignals.length === 0 ? (
              <span className="text-muted-foreground">Zurzeit keine Signale.</span>
            ) : (
              liveSignals.map(({ ticker, signal, price, time }) => {
                const colorClass = signal === "BUY"
                  ? "bg-green-200 text-green-700"
                  : "bg-red-200 text-red-700";

                return (
                  <div
                    key={ticker + time}
                    className={`px-5 py-2 rounded-lg bg-accent shadow hover:scale-105 transition-transform font-mono flex gap-3 items-center`}
                  >
                    <span className="font-semibold">{ticker}</span>
                    <span className={`text-sm px-2 py-0.5 rounded ${colorClass}`}>
                      {signal}
                    </span>
                    <span className="">{price.toFixed(2)} EUR</span>
                    <span className="text-xs text-muted-foreground">{time}</span>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}
