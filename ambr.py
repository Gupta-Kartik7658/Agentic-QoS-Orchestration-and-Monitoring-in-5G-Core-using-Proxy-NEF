import requests
import json
from datetime import datetime
import time

# Use localhost since we're running on the VM
BASE_URL = "http://127.0.0.1:8000"

class Phase4Tester:
    def __init__(self, base_url):
        self.base_url = base_url
        self.log = []
    
    def log_action(self, action, result, status="OK"):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "result": result,
            "status": status
        }
        self.log.append(entry)
        print(f"\n[{entry['timestamp']}] {action}")
        print(f"  Status: {status}")
        print(f"  Result: {json.dumps(result, indent=2) if isinstance(result, dict) else result}")
    
    def get_sessions(self):
        """Fetch all active PDU sessions"""
        try:
            r = requests.get(f"{self.base_url}/core/sessions", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            self.log_action("GET /core/sessions", str(e), "FAIL")
            return None
    
    def get_subscriber(self, imsi):
        """Fetch subscriber profile"""
        try:
            r = requests.get(f"{self.base_url}/core/subscriber/{imsi}", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            self.log_action(f"GET /core/subscriber/{imsi}", str(e), "FAIL")
            return None
    
    def get_policy(self, imsi):
        """Fetch current policy for subscriber"""
        try:
            r = requests.get(f"{self.base_url}/core/policy/{imsi}", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            self.log_action(f"GET /core/policy/{imsi}", str(e), "FAIL")
            return None
    
    def set_ambr_policy(self, imsi, target_policy):
        """Change the QoS policy (which controls AMBR)"""
        try:
            # Send as 'policy_name' to match the endpoint
            payload = {"policy_name": target_policy}
            r = requests.put(f"{self.base_url}/core/subscriber/{imsi}/ambr", json=payload, timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            self.log_action(f"PUT /core/subscriber/{imsi}/ambr to {target_policy}", str(e), "FAIL")
            return None
    
    def test_ambr_policy_switch(self, imsi, policy1="default", policy2="premium"):
        """Test: switch between QoS policies, verify, switch back"""
        print("\n" + "="*70)
        print(f"TEST: QoS Policy Switch for UE {imsi}")
        print("="*70)
        
        # Read baseline
        baseline = self.get_subscriber(imsi)
        self.log_action(f"Read baseline for {imsi}", baseline, "OK")
        
        policy_baseline = self.get_policy(imsi)
        self.log_action(f"Current policy for {imsi}", policy_baseline, "OK")
        
        # Switch to policy2
        result = self.set_ambr_policy(imsi, policy2)
        self.log_action(f"Switch {imsi} from {policy1} to {policy2}", result, "OK" if result else "FAIL")
        
        # Verify switch
        time.sleep(0.5)
        switched = self.get_subscriber(imsi)
        self.log_action(f"Verify policy switch for {imsi}", switched, "OK")
        
        policy_switched = self.get_policy(imsi)
        self.log_action(f"Policy after switch for {imsi}", policy_switched, "OK")
        
        # Switch back
        result = self.set_ambr_policy(imsi, policy1)
        self.log_action(f"Switch {imsi} back to {policy1}", result, "OK" if result else "FAIL")
        
        # Verify restore
        time.sleep(0.5)
        restored = self.get_subscriber(imsi)
        self.log_action(f"Verify restore for {imsi}", restored, "OK")
        
        policy_restored = self.get_policy(imsi)
        self.log_action(f"Policy after restore for {imsi}", policy_restored, "OK")
    
    def test_multiple_ues_independent_policies(self, imsis):
        """Test: assign different policies to different UEs"""
        print("\n" + "="*70)
        print(f"TEST: Multiple UEs with Independent Policies")
        print("="*70)
        
        policies = ["default", "premium", "standard"]
        
        for i, imsi in enumerate(imsis[:3]):
            target_policy = policies[i % len(policies)]
            result = self.set_ambr_policy(imsi, target_policy)
            self.log_action(f"Assign {target_policy} to {imsi}", result, "OK" if result else "FAIL")
            time.sleep(0.3)
        
        # Verify all assignments
        for imsi in imsis[:3]:
            policy_data = self.get_policy(imsi)
            self.log_action(f"Verify policy for {imsi}", policy_data, "OK")
    
    def dump_log(self, filename="phase4_test_log.json"):
        """Save all actions and results to a JSON log"""
        with open(filename, 'w') as f:
            json.dump(self.log, f, indent=2)
        print(f"\n\nTest log saved to {filename}")

# Run the tests
if __name__ == "__main__":
    tester = Phase4Tester(BASE_URL)
    
    # Get current sessions to find real IMSIs
    print("\n" + "="*70)
    print("DISCOVERING ACTIVE UES")
    print("="*70)
    sessions_resp = tester.get_sessions()
    
    if not sessions_resp:
        print("ERROR: Could not fetch sessions.")
        exit(1)
    
    # Fix: Your proxy returns 'active_sessions' not 'sessions'
    sessions = sessions_resp.get('active_sessions', [])
    
    if not sessions:
        print("ERROR: No active sessions found. Make sure UEs are connected via UERANSIM.")
        exit(1)
    
    # Extract IMSIs from sessions
    imsis = list(set([s.get('imsi') for s in sessions if s.get('imsi')]))
    print(f"\nFound {len(imsis)} UEs: {imsis}")
    
    if not imsis:
        print("ERROR: Could not extract IMSIs from sessions.")
        exit(1)
    
    # Test 1: Policy switch on first UE
    if imsis:
        tester.test_ambr_policy_switch(imsis[0], "default", "premium")
    
    # Test 2: Multiple UEs with different policies
    if len(imsis) >= 2:
        tester.test_multiple_ues_independent_policies(imsis)
    
    # Dump final log
    tester.dump_log()
    
    print("\n\n" + "="*70)
    print("PHASE 4 TESTS COMPLETE")
    print("="*70)
    print(f"Detailed log saved to phase4_test_log.json")