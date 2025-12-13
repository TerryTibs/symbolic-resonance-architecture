import React from 'react';
import { roadmapData } from './constants';
import Header from './components/Header';
import RoadmapStage from './components/RoadmapStage';

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-slate-900 font-sans p-4 sm:p-6 md:p-8">
      <div className="max-w-7xl mx-auto">
        <Header />
        <main className="mt-12 space-y-16">
          {roadmapData.map((stage) => (
            <RoadmapStage key={stage.stage} stageData={stage} />
          ))}
        </main>
        <footer className="text-center mt-20 pb-8 text-slate-500">
          <p>Inspired by the cognitive patterns in the Gospel of Thomas.</p>
          <p>A conceptual blueprint for a self-reflective AI.</p>
        </footer>
      </div>
    </div>
  );
};

export default App;
