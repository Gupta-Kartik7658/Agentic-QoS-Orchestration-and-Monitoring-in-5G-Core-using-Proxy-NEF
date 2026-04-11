from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import httpx
import datetime
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Ella-Core NEF Proxy", description="Agentic API for 5G QoS Orchestration")

ELLA_CORE_URL = os.getenv("ELLA_CORE_URL", "http://127.0.0.1:5002/api/v1")
ELLA_API_TOKEN = os.getenv("ELLA_CORE_TOKEN")

headers = {
    "Authorization": f"Bearer {ELLA_API_TOKEN}",
    "Content-Type": "application/json"
}

agent_logs: List[Dict[str, Any]] = []

class PolicyUpdateRequest(BaseModel):
    policy_name: str = Field(..., description="Name of the policy to assign")

class LogEntry(BaseModel):
    action: str
    reasoning: str
    imsi: Optional[str] = None
    status: str = "success"

@app.get("/core/status")
async def get_status():
    """Get core health and status."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{ELLA_CORE_URL}/status", headers=headers, timeout=5)
            resp.raise_for_status()
            return resp.json().get("result", {})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/core/metrics")
async def get_metrics():
    """Fetch Prometheus metrics snapshot."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{ELLA_CORE_URL}/metrics", headers=headers, timeout=5)
            resp.raise_for_status()
            return {"metrics": resp.text}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/core/sessions")
async def get_active_sessions():
    """List all registered subscribers (active PDU sessions)."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{ELLA_CORE_URL}/subscribers", headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json().get("result", {}).get("items", [])
            active_sessions = [sub for sub in data if sub.get("status", {}).get("registered")]
            return {"active_sessions": active_sessions}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/core/sessions/{imsi}")
async def get_subscriber_session(imsi: str):
    """Get detailed session info for a specific UE."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{ELLA_CORE_URL}/subscribers/{imsi}", headers=headers, timeout=5)
            if resp.status_code == 404:
                raise HTTPException(status_code=404, detail="Subscriber not found")
            resp.raise_for_status()
            return resp.json().get("result", {})
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)

@app.get("/core/subscriber/{imsi}")
async def get_subscriber_profile(imsi: str):
    """Alias for session detail in Ella-Core."""
    return await get_subscriber_session(imsi)

@app.get("/core/policy/{imsi}")
async def get_subscriber_policy(imsi: str):
    """Fetch the QoS policy currently assigned to the subscriber."""
    try:
        sub_data = await get_subscriber_session(imsi)
        policy_name = sub_data.get("policyName")
        
        if not policy_name:
            raise HTTPException(status_code=404, detail="No policy assigned to this subscriber")

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{ELLA_CORE_URL}/policies/{policy_name}", headers=headers, timeout=5)
            resp.raise_for_status()
            return {"imsi": imsi, "policy": resp.json().get("result", {})}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/core/subscriber/{imsi}/ambr")
async def update_subscriber_ambr(imsi: str, req: PolicyUpdateRequest):
    """
    Update subscriber's QoS policy (AMBR).
    Request body: {"policy_name": "premium"}
    """
    print(imsi)
    policy_name = req.policy_name.strip()
    print(policy_name)
    
    if not policy_name:
        raise HTTPException(status_code=400, detail="policy_name cannot be empty")
    
    async with httpx.AsyncClient() as client:
        try:
            # Verify policy exists first
            policy_check = await client.get(
                f"{ELLA_CORE_URL}/policies/{policy_name}", 
                headers=headers, 
                timeout=5
            )
            print(policy_check)
            if policy_check.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Policy '{policy_name}' not found in Ella-Core")
            
            # Update subscriber with the policy
            payload = {"imsi": imsi, "policyName": policy_name}
            update_resp = await client.put(
                f"{ELLA_CORE_URL}/subscribers/{imsi}",
                json=payload,
                headers=headers,
                timeout=5
            )
            print(update_resp)
            
            if update_resp.status_code != 200:
                raise HTTPException(
                    status_code=update_resp.status_code,
                    detail=f"Ella-Core update failed: {update_resp.text}"
                )
            
            return {
                "message": f"Successfully updated IMSI {imsi} to policy '{policy_name}'",
                "imsi": imsi,
                "policy": policy_name,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            }
        
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.put("/core/subscriber/{imsi}/slice")
async def update_subscriber_slice(imsi: str, sst: int, sd: str = ""):
    """
    Placeholder for slice steering.
    Ella-Core does not support per-subscriber slice assignment via API.
    """
    return {
        "warning": "Ella-Core does not support per-subscriber slice assignment",
        "message": "Endpoint stubbed for agent compatibility",
        "imsi": imsi,
        "sst": sst,
        "sd": sd
    }

@app.post("/agent/log")
async def create_agent_log(entry: LogEntry):
    """Agent POSTs its reasoning and actions here for audit trail."""
    log_record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        **entry.dict()
    }
    agent_logs.append(log_record)
    return {"message": "Log recorded", "log_id": len(agent_logs)}

@app.get("/agent/log")
async def get_agent_logs():
    """Retrieve the agent's audit trail."""
    return {"logs": agent_logs, "count": len(agent_logs)}