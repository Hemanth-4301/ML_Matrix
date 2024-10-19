import React from "react";
import "../styles/Navbar.css";
import { HashLink } from "react-router-hash-link";
import lg1 from "../images/lg1.png";

const Navbar = () => {
  return (
    <nav>
      <input type="checkbox" id="sidebar-active" />
      <img src={lg1} height={90} width={90} className="ms-5"></img>
      <label htmlFor="sidebar-active" className="open-sidebar-button">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          height="35px"
          viewBox="0 -960 960 960"
          width="35px"
          fill="#e8eaed"
        >
          <path d="M120-240v-80h720v80H120Zm0-200v-80h720v80H120Zm0-200v-80h720v80H120Z" />
        </svg>
      </label>

      <div className="link-container">
        <label htmlFor="sidebar-active" className="close-sidebar-button">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            height="35px"
            viewBox="0 -960 960 960"
            width="35px"
            fill="#e8eaed"
          >
            <path d="m256-200-56-56 224-224-224-224 56-56 224 224 224-224 56 56-224 224 224 224-56 56-224-224-224 224Z" />
          </svg>
        </label>

        <HashLink smooth to="#home" className="home-link">
          Home
        </HashLink>
        <HashLink smooth to="#aboutus">
          About us
        </HashLink>
        <HashLink smooth to="#contactus">
          Contact us
        </HashLink>
      </div>
    </nav>
  );
};

export default Navbar;
