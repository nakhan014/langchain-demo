#!/usr/bin/env python3
"""
Test script to verify that all imports are working correctly
"""

try:
    print("Testing imports...")
    
    # Test core imports
    from langchain_groq import ChatGroq
    print("✅ ChatGroq import successful")
    
    from langchain_core.messages import HumanMessage, AIMessage
    print("✅ HumanMessage and AIMessage import successful")
    
    import streamlit as st
    print("✅ Streamlit import successful")
    
    from dotenv import load_dotenv
    print("✅ dotenv import successful")
    
    print("\n🎉 All imports successful! The ModuleNotFoundError has been fixed.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)