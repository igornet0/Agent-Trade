from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse

from core.database import User
from core.database.orm_query import orm_get_coin_by_name
from app.configuration import Server, CoinResponse

import logging

router = APIRouter(tags=["Main"])

logger = logging.getLogger("app_fastapi.main")

@router.get("/")
async def read_root(request: Request):
    return RedirectResponse(url=f"{Server.frontend_url}/")

@router.get("/team_page")
async def read_root(request: Request):
    return RedirectResponse(url=f"{Server.frontend_url}/team_page")

@router.get("/contact_page")
async def read_root(request: Request):
    return RedirectResponse(url=f"{Server.frontend_url}/contact_page")

@router.get("/faq")
async def read_root(request: Request):
    return RedirectResponse(url=f"{Server.frontend_url}/faq")

@router.get("/profile_page")
async def read_root():
    return RedirectResponse(url=Server.frontend_url)

@router.get("/pricing_page")
async def read_root(request: Request):
    return RedirectResponse(url=f"{Server.frontend_url}/pricing_page")