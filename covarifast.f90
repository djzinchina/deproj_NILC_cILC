!######################################################################
! This file is a part of CMBframe
!
! Cosmic Microwave Background (data analysis) frame(work)
! Copyright (C) 2021  Shamik Ghosh
!
! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <https://www.gnu.org/licenses/>.
!
! For more information about CMBframe please visit 
! <https://github.com/1cosmologist/CMBframe> or contact Shamik Ghosh 
! at shamik@ustc.edu.cn
!
!########################################################################

subroutine nbr2idx(n2spix_map_1, nbr_grp_1, npix_1, ngrp_1, idx_map_1)
!======================================================================
! f2py compiled function documentation:
! idx_map_1 = nbr2idx(n2spix_map_1,nbr_grp_1,[npix_1,ngrp_1])
!======================================================================
    implicit none

    integer (kind=4), intent(in) :: ngrp_1
    integer (kind=8), intent(in) :: npix_1
    integer (kind=4), dimension(0:npix_1-1), intent(in) :: n2spix_map_1
    integer (kind=4), dimension(0:ngrp_1-1), intent(in) :: nbr_grp_1
    integer (kind=1), dimension(0:npix_1-1), intent(out) :: idx_map_1

    integer (kind=4) :: spix_1
    integer (kind=8) :: ipix_1!, l
    
    ! write(*,*) npix, ngrp
    ! l = 0
    ! write(*,*) sum(nbr_grp_1), size(nbr_grp_1)
    idx_map_1 = 0

    do spix_1=0, ngrp_1-1
        ! write(*,*) nbr_grp(spix)
        if (nbr_grp_1(spix_1) .GE. 0) then
            ! write(*,*) 'Inside_loop'
            ! l = l+1

!$OMP PARALLEL SHARED(npix_1, n2spix_map_1, nbr_grp_1, idx_map_1) PRIVATE(ipix_1)
!$OMP DO
            do ipix_1=0, npix_1-1
                if (n2spix_map_1(ipix_1) .eq. nbr_grp_1(spix_1)) then
                    idx_map_1(ipix_1) = 1
                    ! write(*,*) ipix_1, idx_map_1(ipix_1)
                end if
            end do 
!$OMP END DO
!$OMP END PARALLEL            
        end if
        ! write(*,*) spix_1, sum(idx_map_1)
    end do 
    ! write(*,*) sum(idx_map_1) !l,(0:npix_1/2) 

end subroutine nbr2idx

subroutine spix2idx(n2spix_map_2, spix_2, npix_2, idx_map_2)
!======================================================================
! f2py compiled function documentation:
! idx_map_2 = spix2idx(n2spix_map_2,spix_2,[npix_2])
!======================================================================
    implicit none

    integer (kind=4), intent(in) :: spix_2
    integer (kind=8), intent(in) :: npix_2
    integer (kind=4), dimension(0:npix_2-1), intent(in) :: n2spix_map_2
    integer (kind=1), dimension(0:npix_2-1), intent(out) :: idx_map_2

    integer (kind=8) :: ipix_2
    
    idx_map_2 = 0

!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ipix_2)
    do ipix_2=0, npix_2-1
        if (n2spix_map_2(ipix_2) .eq. spix_2) then
            idx_map_2(ipix_2) = 1
        end if
    end do
!$OMP END PARALLEL DO
end subroutine spix2idx 

subroutine fast_cov(map_1, map_2, mask, npix_3, cov)
!======================================================================
! f2py compiled function documentation:
! cov = fast_cov(map_1,map_2,mask,[npix_3])
!======================================================================
    implicit none

    integer (kind=8), intent(in) :: npix_3
    real (kind=8), dimension(0:npix_3-1), intent(in) :: map_1, map_2
    integer (kind=1), dimension(0:npix_3-1), intent(in) :: mask
    real (kind=8), intent(out) :: cov

    cov = sum(map_1 * map_2 * dble(mask))

end subroutine fast_cov

subroutine spmat2npmat(s2npix_map, spix_mat, npix_mat, npix_sup, npix_map, nu_dim)
!======================================================================
! f2py compiled function documentation:
! npix_mat = spmat2npmat(s2npix_map,spix_mat,[npix_sup,npix_map,nu_dim])
!======================================================================
    implicit none

    integer (kind=4), intent(in) :: npix_sup, nu_dim
    integer (kind=8), intent(in) :: npix_map  
    integer (kind=4), dimension(0:npix_map-1), intent(in) :: s2npix_map
    real (kind=8), dimension(0:npix_sup-1, 0:nu_dim-1, 0:nu_dim-1), intent(in) :: spix_mat
    real (kind=8), dimension(0:npix_map-1, 0:nu_dim-1, 0:nu_dim-1), intent(out) :: npix_mat

    integer (kind=4) :: spix_3
    integer (kind=8) :: ipix_3

    npix_mat = 0.0d0 

!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(spix_3, ipix_3) 
    do spix_3 = 0, npix_sup-1
        do ipix_3 = 0, npix_map-1
            if (s2npix_map(ipix_3) .eq. spix_3) then
                npix_mat(ipix_3,:,:) = spix_mat(spix_3,:,:)
            end if
        end do  
    end do 
!$OMP END PARALLEL DO
end subroutine spmat2npmat


subroutine compute_cov_mat(npix_sup, npix_map, nu_dim, ngrp, spix_groups, s2npix_map, wavelet_maps, cov_map)
!================================================================================================
! f2py compiled function documentation:
! cov_map = compute_cov_mat(spix_groups,s2npix_map,wavelet_maps,[npix_sup,npix_map,nu_dim,ngrp])
!================================================================================================
    implicit none

    integer (kind=4), intent(in) :: npix_sup, ngrp, nu_dim
    integer (kind=8), intent(in) :: npix_map
    integer (kind=4), dimension(0:npix_sup-1, 0:ngrp-1), intent(in) :: spix_groups
    integer (kind=4), dimension(0:npix_map-1), intent(in) :: s2npix_map
    real (kind=8), dimension(0:nu_dim-1, 0:npix_map-1), intent(in) :: wavelet_maps
    real (kind=8), dimension(0:npix_sup-1, 0:nu_dim-1, 0:nu_dim-1), intent(out) :: cov_map

    integer (kind=4) :: spix, nu_1, nu_2
    real (kind=8) :: cov_loc, cov_pix
! Note OMP_STACKSIZE limit forces maximum size of private variables that can be stored in stack.
! Since mask variables below should be either 1 or zero we can simply set kind = 2 or 1 to fit in the stack size
    integer (kind=4), dimension(:), allocatable :: spix_grp_ith
    integer (kind=1), dimension(:), allocatable :: mask_loc_ith
    integer (kind=1), dimension(:), allocatable :: mask_pix_ith
    real (kind=8), dimension(:), allocatable :: map_1, map_2
    real (kind=8), dimension(0:npix_sup-1, 0:nu_dim-1, 0:nu_dim-1) :: cov_map_sup


    cov_map_sup = 0.

    ! write(*,*) npix_map, npix_sup, nu_dim, ngrp

!$OMP PARALLEL DO DEFAULT(SHARED), PRIVATE(spix, spix_grp_ith, mask_loc_ith, mask_pix_ith), &
!$OMP& PRIVATE(nu_1, nu_2, map_1, map_2, cov_loc, cov_pix)
    do spix=0, npix_sup-1

        allocate(spix_grp_ith(0:ngrp-1))
        spix_grp_ith = 0
        allocate(mask_loc_ith(0:npix_map-1))
        ! mask_loc_ith = 0
        spix_grp_ith(:) = spix_groups(spix,:)
        ! write(*,*) sum(dble(spix_grp_ith))
        call nbr2idx(s2npix_map, spix_grp_ith, npix_map, ngrp, mask_loc_ith)
        deallocate(spix_grp_ith)

        allocate(mask_pix_ith(0:npix_map-1))
        ! mask_pix_ith = 0
        call spix2idx(s2npix_map, spix, npix_map, mask_pix_ith)


        ! write(*,*) sum(dble(mask_pix_ith)), sum(dble(mask_loc_ith))

!OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(nu_1, nu_2, map_1, map_2, cov_loc, cov_pix)
        do nu_1=0, nu_dim-1
            allocate(map_1(0:npix_map-1))
            map_1 = wavelet_maps(nu_1,:)
            do nu_2=nu_1, nu_dim-1
                allocate(map_2(0:npix_map-1))
                map_2 = wavelet_maps(nu_2,:)

                call fast_cov(map_1, map_2, mask_loc_ith, npix_map, cov_loc)
                call fast_cov(map_1, map_2, mask_pix_ith, npix_map, cov_pix)
                cov_map_sup(spix, nu_1, nu_2) = cov_loc - cov_pix
                cov_map_sup(spix, nu_2, nu_1) = cov_map_sup(spix, nu_1, nu_2)
                deallocate(map_2)
            end do 
            deallocate(map_1)
        end do
!OMP END PARALLEL DO
        deallocate(mask_loc_ith, mask_pix_ith)
    end do 
!$OMP END PARALLEL DO

    cov_map = cov_map_sup
    ! call spmat2npmat(s2npix_map,cov_map_sup, cov_map, npix_sup, npix_map, nu_dim)
end subroutine compute_cov_mat

! subroutine compute_pix_mask(npix_sup, npix_map, ngrp, spix_groups, s2npix_map, mask_spix)
!     integer (kind=4), intent(in) :: npix_sup, ngrp
!     integer (kind=8), intent(in) :: npix_map
!     integer (kind=4), dimension(0:npix_sup-1, 0:ngrp-1), intent(in) :: spix_groups
!     integer (kind=4), dimension(:), allocatable :: spix_ith
!     integer (kind=4), dimension(0:npix_map-1), intent(in) :: s2npix_map
! ! Note OMP_STACKSIZE limit forces maximum size of private variables that can be stored in stack.
! ! Since mask variables below should be either 1 or zero we can simply set kind = 2 or 1 to fit in the stack size
!     integer (kind=1), dimension(:), allocatable :: mask_ith
!     integer (kind=2), dimension(0:npix_sup-1, 0:npix_map-1), intent(out) :: mask_spix

!     integer (kind=4) :: spix!,i
!     mask_spix = 0

! !OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(spix,  mask_ith)
! !spix_ith,
!     do spix=0, npix_sup-1
!         allocate(spix_ith(0:ngrp-1))
!         spix_ith = 0
!         allocate(mask_ith(0:npix_map-1))
!         mask_ith = 0
!         ! write(*,*) sum(spix_groups(spix,:))
!         spix_ith(:) = spix_groups(spix,:)
!         call nbr2idx(s2npix_map, spix_ith, npix_map, ngrp, mask_ith)
!         ! call spix2idx(s2npix_map, spix, npix_map, mask_ith)
!         mask_spix(spix,:) = mask_ith(:)
!         ! write(*,*) spix, sum(dble(mask_ith))

!         deallocate(spix_ith,mask_ith) !

!     end do 
! !OMP END PARALLEL DO


! end subroutine compute_pix_mask