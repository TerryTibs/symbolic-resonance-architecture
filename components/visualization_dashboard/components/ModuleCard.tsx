import React from 'react';
import { SraModule, ModuleStatus } from '../types';

interface ModuleCardProps {
  module: SraModule;
}

const statusStyles: { [key in ModuleStatus]: string } = {
    [ModuleStatus.IMPLEMENTABLE_TODAY]: 'bg-green-500/10 text-green-400 border-green-500/30',
    [ModuleStatus.ADVANCED_PROTOTYPE]: 'bg-sky-500/10 text-sky-400 border-sky-500/30',
    [ModuleStatus.RESEARCH_PROTOTYPE]: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    [ModuleStatus.CORE_INNOVATION]: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
};


const ModuleCard: React.FC<ModuleCardProps> = ({ module }) => {
  return (
    <div className="bg-slate-800/50 rounded-lg border border-slate-700/50 shadow-lg p-6 flex flex-col h-full transition-all duration-300 hover:border-cyan-400/50 hover:shadow-cyan-500/10 hover:-translate-y-1">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-4">
           {module.icon}
          <div>
            <h3 className="text-xl font-bold text-slate-100">{module.name}</h3>
            <p className="text-sm font-mono text-cyan-400">{module.acronym}</p>
          </div>
        </div>
      </div>

      <p className="text-slate-400 text-sm mb-6 flex-grow">{module.description}</p>

      <div className="space-y-4 text-xs">
         <div>
            <h4 className="font-semibold text-slate-300 mb-1">Based On Existing Tech:</h4>
            <p className="text-slate-400">{module.existingTech}</p>
         </div>
         <div>
            <h4 className="font-semibold text-rose-400 mb-1">Novel Aspect / Core Innovation:</h4>
            <p className="text-slate-400">{module.novelAspect}</p>
         </div>
      </div>
      
      <div className="mt-6 pt-4 border-t border-slate-700/50">
          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${statusStyles[module.status]}`}>
            {module.status}
          </span>
      </div>

    </div>
  );
};

export default ModuleCard;
