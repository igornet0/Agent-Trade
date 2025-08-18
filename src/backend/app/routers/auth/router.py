from jose import jwt, JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, HTTPException, Depends, status, Body, Request
from fastapi.security import (HTTPBearer,
                              OAuth2PasswordRequestForm)
from datetime import timedelta

from core import settings
from core.database.orm.users import orm_get_user_by_email, orm_get_user_by_login, orm_add_user

from backend.app.configuration import (Server, 
                                       get_password_hash,
                                       UserResponse, UserLoginResponse,
                                       Token,
                                       verify_password, is_email,
                                       create_access_token,
                                       verify_authorization)
from backend.app.configuration.auth import create_refresh_token

import logging

http_bearer = HTTPBearer(auto_error=False)

router = APIRouter(prefix="/auth", tags=["auth"], dependencies=[Depends(http_bearer)])

logger = logging.getLogger("app_fastapi.auth")


@router.post("/register/", response_model=Token)
async def register(user: UserLoginResponse = Body(), session: AsyncSession = Depends(Server.get_db)):

    db_user = await orm_get_user_by_email(session, user)

    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    db_user = await orm_get_user_by_login(session, user)

    if db_user:
        raise HTTPException(status_code=400, detail="Login already registered")
    
    hashed_password = get_password_hash(user.password)

    await orm_add_user(session, login=user.login,
                              hashed_password=hashed_password,
                              email=user.email)
    
    access_token_expires = timedelta(minutes=settings.security.access_token_expire_minutes)

    access_token = create_access_token(payload={"sub": user.login, "email": user.email}, 
                                       expires_delta=access_token_expires)
    refresh_token = create_refresh_token(payload={"sub": user.login, "email": user.email})

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "message": "User registered successfully"
    }


@router.post("/login_user/", response_model=Token)
async def login_for_access_token(
    request: Request,
    session: AsyncSession = Depends(Server.get_db),
):
    
    unauthed_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Try to get data from JSON body first
    try:
        body = await request.json()
        username = body.get("email") or body.get("login") or body.get("username")
        password = body.get("password")
    except:
        # If JSON parsing fails, try form data
        form_data = await request.form()
        username = form_data.get("username") or form_data.get("email") or form_data.get("login")
        password = form_data.get("password")

    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Username/email and password are required"
        )

    identifier_type = "email" if is_email(username) else "login"
    
    if identifier_type == "email":
        user = await orm_get_user_by_email(session, UserLoginResponse(email=username, password=password))
    else:
        user = await orm_get_user_by_login(session, UserLoginResponse(login=username, password=password))

    if not user:
        raise unauthed_exc

    if not verify_password(password, user.password):
        raise unauthed_exc

    if not user.active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )

    access_token_expires = timedelta(minutes=settings.security.access_token_expire_minutes)
    access_token = create_access_token(payload={"sub": user.login, "email": user.email}, 
                                        expires_delta=access_token_expires)
    refresh_token = create_refresh_token(payload={"sub": user.login, "email": user.email})

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        message="User logged in successfully"
    )


@router.post("/refresh-token/", response_model=Token)
async def refresh_token_endpoint(refresh_token: str):
    try:
        payload = jwt.decode(refresh_token, settings.security.refresh_secret_key, algorithms=[settings.security.algorithm])
        token_type = payload.get("type")
        username: str = payload.get("sub")
        if username is None or token_type != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    access_token_expires = timedelta(minutes=settings.security.access_token_expire_minutes)
    new_access_token = create_access_token(payload={"sub": username}, expires_delta=access_token_expires)
    new_refresh_token = create_refresh_token(payload={"sub": username})
    
    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }

@router.get("/user/me/", response_model=UserResponse)
async def auth_user_check_self_info(
    user: str = Depends(verify_authorization)
):
    return user
