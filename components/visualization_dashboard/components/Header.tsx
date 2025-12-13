import React from 'react';

const Header: React.FC = () => {
  return (
    <header className="text-center space-y-4">
      <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-violet-500">
        Symbolic Resonance Architecture
      </h1>
      <p className="text-lg text-slate-400 max-w-3xl mx-auto">
        A practical roadmap for prototyping a novel AI architecture. This system is designed to learn through resonance, self-revelation, and contradiction-driven synthesis.
      </p>
    </header>
  );
};

export default Header;
