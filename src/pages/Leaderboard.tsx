
import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Table, TableHeader, TableBody, TableRow, TableCell, TableHead } from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";

type ModelRow = {
  name: string;
  freq: string;
  label: string;
  family: string;
  score: number;
  sharpe: number;
  bacc: number;
};

const rows: ModelRow[] = [
  { name: "LGBM_1d_B1", freq: "1d", label: "B1", family: "LightGBM", score: 0.72, sharpe: 2.15, bacc: 0.81 },
  { name: "PatchTST_1h_R", freq: "1h", label: "R", family: "PatchTST", score: 0.68, sharpe: 1.98, bacc: 0.74 },
  { name: "TFT_1min_B1", freq: "1min", label: "B1", family: "TFT", score: 0.77, sharpe: 2.4, bacc: 0.85 },
];

const columns = [
  { key: "name", label: "Modell" },
  { key: "freq", label: "Freq" },
  { key: "label", label: "Label" },
  { key: "family", label: "Familie" },
  { key: "score", label: "final_score" },
  { key: "sharpe", label: "Sharpe" },
  { key: "bacc", label: "Balanced-Acc" },
];

export default function Leaderboard() {
  const [sort, setSort] = useState<{ key: keyof ModelRow, order: "asc" | "desc" }>({ key: "score", order: "desc" });

  const sorted = [...rows].sort((a, b) =>
    sort.order === "asc"
      ? (a[sort.key] > b[sort.key] ? 1 : -1)
      : (a[sort.key] < b[sort.key] ? 1 : -1)
  );

  return (
    <div className="flex flex-col items-center max-w-6xl mx-auto pt-10 animate-fade-in">
      <Card className="p-6 w-full shadow-md">
        <h1 className="text-xl font-semibold mb-4">Model Leaderboard</h1>
        <div className="overflow-x-auto rounded-lg">
          <Table>
            <TableHeader>
              <TableRow>
                {columns.map(col => (
                  <TableHead
                    key={col.key}
                    className="cursor-pointer select-none hover:text-primary transition"
                    onClick={() => setSort({
                      key: col.key as keyof ModelRow,
                      order: sort.key === col.key && sort.order === "desc" ? "asc" : "desc"
                    })}
                  >
                    {col.label}
                    {sort.key === col.key && (
                      <span className="ml-1 text-xs">
                        {sort.order === "desc" ? "↓" : "↑"}
                      </span>
                    )}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {sorted.length === 0
                ? <TableRow><TableCell colSpan={columns.length}><Skeleton className="h-6 w-full" /></TableCell></TableRow>
                : sorted.map(row => (
                  <TableRow
                    key={row.name}
                    className="hover:bg-muted/60 transition cursor-pointer"
                  >
                    <TableCell>{row.name}</TableCell>
                    <TableCell>{row.freq}</TableCell>
                    <TableCell>{row.label}</TableCell>
                    <TableCell>{row.family}</TableCell>
                    <TableCell className={`font-bold ${row.score >= 0.7 ? "text-green-700" : row.score < 0.6 ? "text-red-600" : ""}`}>
                      {row.score.toFixed(2)}
                    </TableCell>
                    <TableCell>{row.sharpe.toFixed(2)}</TableCell>
                    <TableCell>{row.bacc.toFixed(2)}</TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        </div>
      </Card>
    </div>
  );
}
