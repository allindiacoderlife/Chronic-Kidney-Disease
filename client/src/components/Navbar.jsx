import React from "react";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

const Navbar = () => {
  useGSAP(() => {
    // Hide navbar when scrolling past the first section (Hero)
    ScrollTrigger.create({
      trigger: ".intro",
      start: "bottom top",
      end: "+=1",
      onEnter: () => {
        gsap.to(".navbar", {
          y: -100,
          opacity: 0,
          duration: 0.3,
          ease: "power2.out",
        });
      },
      onLeaveBack: () => {
        gsap.to(".navbar", {
          y: 0,
          opacity: 1,
          duration: 0.3,
          ease: "power2.out",
        });
      },
    });
  }, []);

  return (
    <nav className="navbar fixed top-0 left-0 z-50 w-full flex flex-row justify-between items-center py-4 px-8 text-white">
      <div className="logo">CKD Predictor</div>
      <ul className="nav-links flex flex-row gap-6">
        <li>
          <a href="#home">Home</a>
        </li>
        <li>
          <a href="#about">About</a>
        </li>
        <li>
          <a href="#contact">Contact</a>
        </li>
      </ul>
      <button>
        <a href="#predict">Predict Now</a>
      </button>
    </nav>
  );
};

export default Navbar;
