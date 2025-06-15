
import { NavLink } from "react-router-dom";
import { Monitor, Table, ChartBar, FileText } from "lucide-react";

const navItems = [
  { name: "Live", icon: Monitor, to: "/" },
  { name: "Leaderboard", icon: Table, to: "/leaderboard" },
  { name: "Equity", icon: ChartBar, to: "/equity" },
  { name: "Explain", icon: FileText, to: "/explain" },
];

export default function DashboardNav() {
  return (
    <nav className="flex w-full px-4 py-2 gap-2 border-b border-border bg-background sticky top-0 z-40 shadow-sm">
      <ul className="flex flex-row gap-2 w-full justify-center">
        {navItems.map(({ name, icon: Icon, to }) => (
          <li key={name}>
            <NavLink
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-2 px-5 py-2 rounded-md transition-all duration-150 text-muted-foreground hover:text-primary hover:bg-muted/60
                 ${isActive ? "bg-muted text-primary font-semibold shadow-inner" : ""}`
              }
            >
              <Icon size={20} className="mr-1" />
              <span className="tracking-wide">{name}</span>
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}
