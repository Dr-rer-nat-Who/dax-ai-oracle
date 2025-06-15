
import DashboardNav from "@/components/DashboardNav";
import Live from "./Live";

const Index = () => {
  return (
    <div className="bg-background min-h-screen w-full text-foreground">
      <DashboardNav />
      <main className="max-w-screen-xl mx-auto">
        <Live />
      </main>
    </div>
  );
};

export default Index;
