"""
Simple test script to verify Flask application is working using urllib
"""
import urllib.request
import urllib.error

def test_app():
    """Test if the Flask app is running"""
    try:
        # Test the main page
        response = urllib.request.urlopen('http://127.0.0.1:5000/', timeout=5)
        if response.getcode() == 200:
            print("✅ Flask application is running successfully!")
            print("✅ Main page is accessible")
        else:
            print(f"❌ Unexpected status code: {response.getcode()}")
            return False
            
        # Test the demo page
        response = urllib.request.urlopen('http://127.0.0.1:5000/demo', timeout=5)
        if response.getcode() == 200:
            print("✅ Demo page is accessible")
        else:
            print(f"❌ Demo page error: {response.getcode()}")
            
        return True
        
    except urllib.error.URLError as e:
        print("❌ Cannot connect to Flask application")
        print(f"   Error: {e}")
        print("   Make sure the app is running with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error testing application: {e}")
        return False

if __name__ == "__main__":
    print("Testing Flask application...")
    print("=" * 40)
    
    if test_app():
        print("\n🎉 All tests passed! Your Flask application is working correctly.")
        print("\nYou can now access the application in your browser at:")
        print("  http://127.0.0.1:5000")
        print("  http://192.168.1.108:5000")
    else:
        print("\n❌ Tests failed. Please check the application.")