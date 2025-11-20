"""
Train all ML models for the hotel booking assistant.
Run this script ONCE before using the chatbot.

Usage:
    python train_models.py
"""

from brain import train_all

if __name__ == "__main__":
    print("=" * 70)
    print("🏋️ Training Hotel Booking ML Models")
    print("=" * 70)
    print("\nThis will train 3 models:")
    print("  1. is_canceled (classification)")
    print("  2. total_stay_nights (regression)")
    print("  3. deposit_type (classification)")
    print("\nThis may take a few minutes...\n")

    try:
        train_all("hotel_bookings.csv")
        print("\n" + "=" * 70)
        print("✅ All models trained successfully!")
        print("=" * 70)
        print("\nModel files created:")
        print("  • model_is_canceled.pkl")
        print("  • model_total_stay_nights.pkl")
        print("  • model_deposit_type.pkl")
        print("\n🚀 You can now run the chatbot: python chatbot.py")

    except FileNotFoundError:
        print("\n❌ Error: hotel_bookings.csv not found!")
        print("Make sure the CSV file is in the same directory.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback

        traceback.print_exc()