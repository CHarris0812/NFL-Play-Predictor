import { useState } from "react";
import "./App.css";
import PredictTab from "./PredictTab";
import ReplayTab from "./ReplayTab";

const TABS = [
  { key: "predict", label: "Predict", component: PredictTab },
  { key: "replay", label: "Replay a game", component: ReplayTab },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("predict");
  const ActiveComponent = TABS.find((t) => t.key === activeTab).component;

  return (
    <div className="page">
      <h1>NFL Play Predictor</h1>

      <div className="tabs">
        {TABS.map(({ key, label }) => (
          <button
            key={key}
            className={key === activeTab ? "tab tab-active" : "tab"}
            onClick={() => setActiveTab(key)}
          >
            {label}
          </button>
        ))}
      </div>

      <ActiveComponent />
    </div>
  );
}
