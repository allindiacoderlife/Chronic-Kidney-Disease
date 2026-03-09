import React from "react";

const Hero = () => {
  return (
    <section className="intro">
      <div className="space-y-4">
        <h1 className="text-white font-bold text-5xl lg:text-7xl leading-tight uppercase tracking-tight mt-20">
          Chronic Kidney Disease <br /> Data Preprocessing
        </h1>
        <p className="text-white/50 text-xl font-medium">Predictive Model</p>
      </div>
      <div className="bg-white/10 backdrop-blur-sm border border-white/20 w-1/2 h-20 rounded-full justify-between items-center flex px-6 gap-4 hover:bg-white/20 transition-all duration-300">
        <p className="text-white/80 font-medium">Scroll Down</p>
        <div className="bg-white size-14 rounded-full flex items-center justify-center">
          <svg className="w-6 h-6 text-black animate-bounce" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
        </div>
      </div>
    </section>
  );
};

export default Hero;
