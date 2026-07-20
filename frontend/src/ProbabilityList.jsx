export default function ProbabilityList({ title, probabilities }) {
  if (!probabilities) return null;

  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="result-panel">
      <h3>{title}</h3>
      {sorted.map(([label, prob]) => (
        <div className="bar-row" key={label}>
          <span className="bar-label">{label}</span>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${prob * 100}%` }} />
          </div>
          <span className="bar-value">{(prob * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}
