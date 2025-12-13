import React, { useState } from 'react';
import { RoadmapStageData } from '../types';
import ModuleCard from './ModuleCard';
import PrototypeViewer from './PrototypeViewer';
import { PlayIcon } from '../constants';

interface RoadmapStageProps {
  stageData: RoadmapStageData;
}

const RoadmapStage: React.FC<RoadmapStageProps> = ({ stageData }) => {
  const [prototypeToShow, setPrototypeToShow] = useState<number | null>(null);

  const canShowPrototype = stageData.stage >= 1 && stageData.stage <= 5;

  return (
    <section>
      <div className="relative mb-8">
        <div className="absolute inset-0 flex items-center" aria-hidden="true">
          <div className="w-full border-t border-slate-700" />
        </div>
        <div className="relative flex justify-start">
          <span className="bg-slate-900 pr-3 text-2xl font-semibold text-slate-300">
            Stage {stageData.stage}: <span className="text-cyan-400">{stageData.title}</span>
          </span>
        </div>
      </div>
      <p className="text-slate-400 mb-6 max-w-4xl">{stageData.description}</p>
      
      {canShowPrototype && (
        <div className="mb-8">
          <button
            onClick={() => setPrototypeToShow(stageData.stage)}
            className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-cyan-600 hover:bg-cyan-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 focus:ring-offset-slate-900 transition-colors"
          >
            <PlayIcon />
            View Stage {stageData.stage} Prototype Code
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {stageData.modules.map((module) => (
          <ModuleCard key={module.acronym} module={module} />
        ))}
      </div>
      
      {prototypeToShow !== null && (
        <PrototypeViewer 
          stage={prototypeToShow} 
          onClose={() => setPrototypeToShow(null)} 
        />
      )}
    </section>
  );
};

export default RoadmapStage;
